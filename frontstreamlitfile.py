import uuid
import streamlit as st
from langgraph.types import Command

from backstreamlitfile import app, load_balance, ACCOUNT_BALANCE

st.set_page_config(page_title="AI Buyer Agent", page_icon="🛒")
st.title("AI Buyer Agent")
st.caption("Tell it what to buy. It searches, you pick, it pays — with your approval when it matters.")

# ---------- Session state ----------
# Each browser session gets its own LangGraph thread_id, so two people using
# the app at once (or two runs in the same session) don't collide or resume
# from each other's leftover state.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "result" not in st.session_state:
    st.session_state.result = None
if "started" not in st.session_state:
    st.session_state.started = False

config = {"configurable": {"thread_id": st.session_state.thread_id}}


def reset():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.result = None
    st.session_state.started = False


# ---------- Step 1: what to buy ----------
if not st.session_state.started:
    task = st.text_input("What would you like to buy?", placeholder="buy a wireless mouse under 800")
    if st.button("Start", type="primary", disabled=not task.strip()):
        initial_state = {
            "task": task.strip(),
            "spent_so_far": 0,
            "purchase_history": [],
            "conversation_history": [],
        }
        st.session_state.result = app.invoke(initial_state, config=config)
        st.session_state.started = True
        st.rerun()

# ---------- Step 2+: react to wherever the graph currently is ----------
else:
    result = st.session_state.result
    interrupts = result.get("__interrupt__")

    if interrupts:
        payload = interrupts[0].value

        if payload["type"] == "select_item":
            st.subheader("Search results")
            for opt in payload["options"]:
                with st.container(border=True):
                    cols = st.columns([3, 1])
                    cols[0].markdown(f"**{opt['brand']} — {opt['name']}**")
                    cols[0].caption(f"{opt['category']} · {opt['vendor']} · {opt['rating']} / 5")
                    cols[1].markdown(f"**Rs.{opt['price']}**")

            reply = st.text_input("Which one do you want to buy? (type it loosely, that's fine)")
            if st.button("Select", type="primary", disabled=not reply.strip()):
                st.session_state.result = app.invoke(Command(resume=reply.strip()), config=config)
                st.rerun()

        elif payload["type"] == "human_approval":
            item = payload["item"]
            st.subheader("Approval needed")
            with st.container(border=True):
                st.markdown(f"**{item['name']}**")
                st.caption(f"{item['vendor']} · {item['rating']} / 5")
                st.markdown(f"**Rs.{payload['price']}**")
                st.caption(f"Remaining balance: Rs.{load_balance()} (out of Rs.{ACCOUNT_BALANCE})")

            c1, c2 = st.columns(2)
            if c1.button("Approve", type="primary"):
                st.session_state.result = app.invoke(Command(resume="yes"), config=config)
                st.rerun()
            if c2.button("Reject"):
                st.session_state.result = app.invoke(Command(resume="no"), config=config)
                st.rerun()

    else:
        # Graph has finished — show the outcome.
        selected = result.get("selected_option")
        history = result.get("purchase_history", [])

        if history:
            last = history[-1]
            order = last["order_result"]
            item = last["item"]
            st.success(f"Purchased **{item['name']}** from {item['vendor']} — Rs.{item['price']}")
            st.caption(f"Razorpay Order ID: {order.get('id')}")
            st.caption(f"Remaining balance: Rs.{load_balance()} (out of Rs.{ACCOUNT_BALANCE})")
        elif selected is None:
            st.warning("Couldn't find or match a suitable item.")
        else:
            st.info("Purchase was not approved — nothing was bought.")

        if st.button("Start another purchase"):
            reset()
            st.rerun()