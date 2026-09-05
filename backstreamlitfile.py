import os
import uuid
import sqlite3
from typing import TypedDict, Optional
from dotenv import load_dotenv
import json
import razorpay
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

# ---------- 0. Load env vars & create clients ----------
load_dotenv()

razorpay_client = razorpay.Client(
    auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
)

llm = ChatOpenAI(model="gpt-5.6-sol", api_key=os.getenv("OPENAI_API_KEY"))


def _print(*args, **kwargs):
    # Only prints when main.py is run directly from the terminal (CLI mode).
    # When Streamlit imports this file, __name__ is "main", not "__main__",
    # so these calls become silent — nothing spills into the server console,
    # everything the user sees comes from the Streamlit page instead.
    if __name__ == "__main__":
        print(*args, **kwargs)


# ---------- 1. Mock vendor catalog (LLM-generated, not a real marketplace API) ----------
# NOTE: No real marketplace API used here (deliberate scope decision for the
# 2-day build). Instead of a hardcoded list, the LLM generates plausible
# product options on the fly for whatever item the user asks for — so the
# agent isn't limited to a fixed set of categories like "mouse" or "keyboard".
# Extension point: swap this function's body for a real Amazon/Flipkart
# affiliate API call or a paid aggregator API once those credentials exist.


# ---------- 2. State schema ----------
class BuyerState(TypedDict):
    task: str                          # e.g. "buy a wireless mouse under 800"
    item_query: Optional[str]
    budget_limit: float
    spent_so_far: float
    candidate_options: list
    selected_option: Optional[dict]
    purchase_history: list
    user_approval: Optional[bool]
    conversation_history: list


# ---------- 3. understand_task (LLM node) ----------
def understand_task(state: BuyerState) -> BuyerState:
    prompt = f"""
    Task: {state['task']}

    Extract the item to buy and the budget limit (in INR, a plain number).
    If no budget is mentioned in the task, use 999999 to mean "no real limit".
    Respond ONLY with valid JSON, no markdown, no explanation:
    {{"item": "...", "budget": number}}
    """
    result = llm.invoke(prompt)
    cleaned = result.content.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)

    state["item_query"] = parsed["item"]
    state["budget_limit"] = float(parsed["budget"])
    return state


# ---------- 4. search_options (LLM-generated mock catalog) ----------
def search_options(state: BuyerState) -> BuyerState:
    prompt = f"""
    A user wants to buy: "{state['item_query']}"
    Their budget limit is Rs.{state['budget_limit']}.

    Generate 5 realistic, plausible product options for this item, as if they
    came from real Indian e-commerce vendors (Amazon, Flipkart, Meesho, Croma).
    ALL 5 options must have a price at or below Rs.{state['budget_limit']} —
    do not include anything above the budget. Vary the brand, price, and
    rating across options so a comparison is meaningful.

    Respond ONLY with valid JSON, no markdown, no explanation — a JSON array
    of exactly 5 objects, each with this exact structure:
    [
      {{
        "id": "v1",
        "brand": "...",
        "name": "...",
        "category": "...",
        "price": number,
        "rating": number,
        "vendor": "Amazon | Flipkart | Meesho | Croma"
      }}
    ]
    "id" must be v1 through v5. "price" is a plain number in INR (no currency symbol),
    and must not exceed Rs.{state['budget_limit']}.
    "rating" is a number between 3.5 and 4.8.
    """
    result = llm.invoke(prompt)
    cleaned = result.content.strip().replace("```json", "").replace("```", "").strip()
    options = json.loads(cleaned)

    state["candidate_options"] = options

    _print("\n" + "=" * 50)
    _print(f"SEARCH RESULTS for \"{state['item_query']}\"")
    _print("=" * 50)
    for i, opt in enumerate(options, start=1):
        _print(f"\n{i}. {opt['brand']} - {opt['name']}")
        _print(f"   Category : {opt['category']}")
        _print(f"   Vendor   : {opt['vendor']}")
        _print(f"   Rating   : {opt['rating']} / 5")
        _print(f"   Price    : Rs.{opt['price']}")
    _print()

    return state


# ---------- 5. select_item (human picks, LLM matches free-text reply) ----------
def select_item(state: BuyerState) -> BuyerState:
    # interrupt() pauses the graph here and hands this payload to whoever is
    # driving it (CLI or Streamlit). The graph resumes when they call
    # app.invoke(Command(resume=<reply>), config=...). This works the same
    # way regardless of what's on the other end — terminal input() or a
    # Streamlit text box.
    reply = interrupt({
        "type": "select_item",
        "options": state["candidate_options"],
    })

    prompt = f"""
    The user was shown this list of products:
    {json.dumps(state['candidate_options'])}

    The user replied: "{reply}"

    Figure out which product (by "id") the user means, even if they typed it
    loosely, misspelled it, or only gave part of the name/brand.
    Respond ONLY with valid JSON, no markdown:
    {{"chosen_id": "..."}}
    If you genuinely cannot match it to any product, use {{"chosen_id": null}}.
    """
    result = llm.invoke(prompt)
    cleaned = result.content.strip().replace("```json", "").replace("```", "").strip()
    chosen = json.loads(cleaned)

    selected = next(
        (o for o in state["candidate_options"] if o["id"] == chosen.get("chosen_id")),
        None
    )
    state["selected_option"] = selected

    if selected is None:
        _print("\nSorry, couldn't match that to any item in the list above.")

    return state


# ---------- 6. route_after_selection (code, not LLM — read-only routing) ----------
def route_after_selection(state: BuyerState) -> str:
    if state["selected_option"] is None:
        return "no_option_found"
    return "human_approval"


# ---------- 7. no_option_found (terminal node) ----------
def no_option_found(state: BuyerState) -> BuyerState:
    state["conversation_history"].append(
        {"result": "Could not match the user's reply to any listed item."}
    )
    return state


# ---------- 8. human_approval (real terminal input, risk-based) ----------
AUTO_APPROVE_THRESHOLD = 200  # INR — purchases below this are auto-approved
ACCOUNT_BALANCE = 10000       # INR — starting "preferred account" balance
BALANCE_FILE = "account_balance.json"  # persists balance across runs (thread_id resets each run, this doesn't)


def load_balance():
    if os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE) as f:
            return json.load(f)["balance"]
    return ACCOUNT_BALANCE


def save_balance(balance):
    with open(BALANCE_FILE, "w") as f:
        json.dump({"balance": balance}, f)


def human_approval(state: BuyerState) -> BuyerState:
    price = state["selected_option"]["price"]
    remaining_balance = load_balance()

    if price > remaining_balance:
        _print(f"\nInsufficient balance. Remaining balance: Rs.{remaining_balance}, "
              f"item price: Rs.{price}. Order should not proceed.")
        state["user_approval"] = False
        return state

    if price < AUTO_APPROVE_THRESHOLD:
        _print(f"\nPurchase amount (Rs.{price}) is below the auto-approval threshold "
              f"(Rs.{AUTO_APPROVE_THRESHOLD}) — auto-approving.")
        state["user_approval"] = True
        return state

    # Same interrupt() pattern as select_item — pause and wait for a
    # yes/no from whoever is driving the graph.
    decision = interrupt({
        "type": "human_approval",
        "item": state["selected_option"],
        "price": price,
    })
    state["user_approval"] = str(decision).strip().lower() == "yes"
    return state


def route_after_approval(state: BuyerState) -> str:
    return "execute_payment" if state["user_approval"] else "END"


# ---------- 9. execute_payment (tool node) ----------
def execute_payment(state: BuyerState) -> BuyerState:
    price = state["selected_option"]["price"]
    item_name = state["selected_option"]["name"]
    vendor = state["selected_option"]["vendor"]

    order = razorpay_client.order.create({
        "amount": int(price * 100),  # paise — Razorpay always takes amount in paise
        "currency": "INR",
        "notes": {"item": item_name}
    })

    state["spent_so_far"] += price
    state["purchase_history"].append({
        "item": state["selected_option"],
        "order_result": order,
    })
    state["conversation_history"].append({"execution_result": order})

    balance_before = load_balance()  #
    remaining_balance = balance_before - price
    save_balance(remaining_balance)
    state["balance_before_purchase"] = balance_before  #
    
    _print(f"\nYour purchase of \"{item_name}\" from {vendor} is successful. "
          f"(Order ID: {order.get('id')}, Amount: Rs.{price})")
    _print(f"Remaining account balance: Rs.{remaining_balance} (out of Rs.{balance_before})") #

    return state


# ---------- 10. log_and_update_memory (tool node) ----------
def log_and_update_memory(state: BuyerState) -> BuyerState:
    # Keeping this lightweight for the demo — appends to an in-memory
    # purchase_history list (already updated in execute_payment).
    # Extension point: push to a vector store so future runs can reference
    # past purchases.
    return state


# ---------- 11. Graph wiring ----------
graph = StateGraph(BuyerState)

graph.add_node("understand_task", understand_task)
graph.add_node("search_options", search_options)
graph.add_node("select_item", select_item)
graph.add_node("no_option_found", no_option_found)
graph.add_node("human_approval", human_approval)
graph.add_node("execute_payment", execute_payment)
graph.add_node("log_and_update_memory", log_and_update_memory)

graph.set_entry_point("understand_task")
graph.add_edge("understand_task", "search_options")
graph.add_edge("search_options", "select_item")

graph.add_conditional_edges(
    "select_item",
    route_after_selection,
    {
        "human_approval": "human_approval",
        "no_option_found": "no_option_found",
    }
)

graph.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {"execute_payment": "execute_payment", "END": END}
)

graph.add_edge("execute_payment", "log_and_update_memory")
graph.add_edge("log_and_update_memory", END)
graph.add_edge("no_option_found", END)

# ---------- 12. Checkpointer ----------
conn = sqlite3.connect(database="razorpay.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
app = graph.compile(checkpointer=checkpointer)


# ---------- 13. print_summary ----------
def print_summary(state: dict):
    print("\n" + "=" * 50)
    print("SUMMARY - Final result")
    print("=" * 50)
    print(f"Task               : {state.get('task')}")
    print(f"Budget Limit       : Rs.{state.get('budget_limit')}")

    selected = state.get("selected_option")
    if selected:
        print(f"Selected Item      : {selected['name']}")
        print(f"Vendor             : {selected['vendor']}")
        print(f"Price              : Rs.{selected['price']}")
    else:
        print("Selected Item      : None (no suitable option found)")

    print(f"User Approval      : {state.get('user_approval')}")

    history = state.get("purchase_history", [])
    if history:
        last = history[-1]
        order = last["order_result"]
        item = last["item"]
        print("Execution Result   :")
        print(f"   Item           : {item['name']}")
        print(f"   Vendor         : {item['vendor']}")
        print(f"   Amount         : Rs.{item['price']}")
        print(f"   Razorpay Order ID : {order.get('id')}")
    else:
        print("Execution Result   : No purchase executed")

    print(f"Total Spent So Far : Rs.{state.get('spent_so_far', 0)}")
    print(f"Remaining Balance  : Rs.{load_balance()} (out of Rs.{ACCOUNT_BALANCE})")
    print("=" * 50)


# ---------- 14. Running it ----------
def _handle_interrupts_cli(result: dict, config: dict) -> dict:
    """Drives interrupt()-based nodes from the terminal. Same graph, same
    interrupt payloads that Streamlit's UI reacts to — just answered here
    with input() instead of buttons."""
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value

        if payload["type"] == "select_item":
            reply = input("\nWhich one do you want to buy? (type the item/model name, "
                          "you can just paste it loosely): ").strip()
            result = app.invoke(Command(resume=reply), config=config)

        elif payload["type"] == "human_approval":
            item = payload["item"]
            print("\n" + "=" * 50)
            print("HUMAN APPROVAL NEEDED")
            print("=" * 50)
            print(f"Item       : {item['name']}")
            print(f"Vendor     : {item['vendor']}")
            print(f"Price      : Rs.{payload['price']}")
            print(f"Rating     : {item['rating']}")
            decision = input("\nApprove this purchase? (yes/no): ").strip().lower()
            result = app.invoke(Command(resume=decision), config=config)

        else:
            break  # unknown interrupt type — stop rather than loop forever

    return result


if __name__ == "__main__":
    thread_id = "buyer_session_1"  # single fixed thread_id for now, instead of a fresh uuid per run
    config = {"configurable": {"thread_id": thread_id}}

    task_input = input("What would you like to buy? (e.g. 'buy a wireless mouse under 800'): ").strip()

    initial_state = {
        "task": task_input,
        "spent_so_far": 0,
        "purchase_history": [],
        "conversation_history": [],
    }

    final_state = app.invoke(initial_state, config=config)
    final_state = _handle_interrupts_cli(final_state, config)
    print_summary(final_state)