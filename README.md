# AI Buyer Agent - Razorpay

An autonomous buyer agent built with LangChain / LangGraph that understands a purchase task in plain language, generates realistic vendor options, lets the user pick one, and completes payment via Razorpay's test-mode API — with risk-based human approval and a persistent account balance.

Built for **Razorpay AI Buildathon 2026, Track 1 (AI Growth & Agentic Commerce)**.

---

## Problem Statement

Razorpay's Track 1 brief calls out **"enabling end-to-end transactions for AI buyers"** as one of the two core directions for agentic commerce. Today, buying something online is a manual, repetitive loop: search vendors, compare prices, decide, then pay — every single time, for every purchase.

This project builds the other side of that future: not a merchant tool, but an **agent that shops on a human's behalf** — understanding what they want, comparing realistic options, and safely completing the payment through Razorpay, with a human kept in the loop only where the risk genuinely warrants it.

There's a second, equally important problem underneath this: **no AI agent should ever hold unscoped access to a user's full financial account.** An autonomous buyer agent is only trustworthy if it can *only* spend what it's explicitly been allowed to — never more. This is exactly the problem [RazorpayX's Sub-Accounts](https://razorpay.com/docs/x/payouts/sub-accounts/) already solves for businesses: a master account allocates a fixed, static spending limit to a sub-account, and the sub-account holder can never exceed it or touch the master account directly. This project's agent is built around that same principle from the ground up — it operates against a fixed, capped balance, not open-ended access to money. (See *Design Decisions* below for how this is modeled in the current build, and *Future Improvements* for how it maps to a real RazorpayX Sub-Account in production.)

## Agent Behavior — How It Works

1. **Understand the task** — the user states what they want in plain English (e.g. *"buy a wireless mouse under 800"*). An LLM extracts the item and the budget. If no budget is mentioned, the agent treats it as effectively unlimited rather than guessing a number and failing later.
2. **Search options** — since a real marketplace API (Amazon/Flipkart affiliate access, or a paid aggregator) requires business approval that isn't feasible in a 2-day build, the agent uses an LLM to generate 5 realistic, varied product options (brand, price, rating, vendor) for whatever the user asked for. This isn't limited to a fixed set of categories — ask for a mouse, a book, or a power bank, and it adapts.
3. **Select an item** — the results are shown to the user, who replies in free text (loosely typed, misspelled, or partial — "JBL one from amazon" works). An LLM matches that reply back to the correct product from the list.
4. **Risk-based approval** — before any payment:
   - If the item's price exceeds the account's remaining balance, the order is rejected outright ("Insufficient balance — order should not proceed").
   - If the price is below a configurable auto-approve threshold (₹200), the agent proceeds without asking — small, low-risk purchases shouldn't need a human's time.
   - Above that threshold, the agent explicitly asks the user: *"Approve this purchase? (yes/no)"* — real terminal input, not a simulated prompt.
5. **Execute payment** — on approval, the agent calls Razorpay's `order.create()` API directly. This is a real, verifiable API call — the resulting orders are visible in the Razorpay Dashboard (Test Mode → Transactions → Orders) with real order IDs and timestamps.
6. **Track balance** — a persistent account balance (starting at ₹10,000) is stored locally and decremented after every successful purchase, so spending carries over across runs rather than resetting each time.

## Human-in-the-Loop: Why Two Touchpoints, Not More

The agent asks the human exactly twice:
- **What to buy** (the task)
- **Whether to approve** (only for purchases above the auto-approve threshold, or already blocked if the balance can't cover it)

Everything in between — search, comparison, matching the user's reply to a product — is handled by the agent. This was a deliberate design choice: letting the human pick from a numbered list, or approve every purchase regardless of amount, would turn the agent into a glorified search-and-display tool and undercut the "AI Judgment" this track is evaluated on. The two touchpoints that remain are the ones that genuinely need a human — intent, and risk sign-off.

## Architecture

```
understand_task (LLM: extract item + budget)
        │
        ▼
search_options (LLM: generate 5 realistic vendor options)
        │
        ▼
select_item (human free-text reply → LLM matches to a product)
        │
   ┌────┴─────┐
   ▼          ▼
no_option   human_approval (balance check → auto-approve <₹200 → else ask human)
_found          │
   │        ┌───┴────┐
   │        ▼        ▼
   │   execute_payment  END (rejected/insufficient balance)
   │        │
   │        ▼
   │   log_and_update_memory
   │        │
   └────────▼
           END
```

State is persisted via LangGraph's `SqliteSaver` checkpointer (`razorpay.db`), using a single fixed `thread_id` for this demo so the session's conversation history is inspectable across runs.

## Tech Stack

| Component            | Choice                                  |
|-----------------------|------------------------------------------|
| Agent orchestration   | LangGraph                               |
| LLM                    | GPT-4o-mini (via `langchain-openai`)    |
| Payments               | Razorpay Python SDK (test mode)         |
| State persistence      | LangGraph `SqliteSaver` (`razorpay.db`) |
| Balance persistence    | Local JSON file (`account_balance.json`)|

## Setup

**1. Install dependencies:**
```
pip install razorpay langchain-openai langgraph langgraph-checkpoint-sqlite python-dotenv
```

**2. Create a `.env` file:**
```
RAZORPAY_KEY_ID=your_test_key_id
RAZORPAY_KEY_SECRET=your_test_key_secret
OPENAI_API_KEY=your_openai_key
```

**3. Run it:**
```
python main.py
```

You'll be prompted for what to buy, shown 5 generated options, asked to pick one in your own words, and — depending on price and balance — either see an instant purchase or be asked to approve it.

## Example Run

```
What would you like to buy? (e.g. 'buy a wireless mouse under 800'): buy me python handbook under 800

==================================================
SEARCH RESULTS for "python handbook"
==================================================

1. O'Reilly Media - Python Crash Course
   Category : Books
   Vendor   : Amazon
   Rating   : 4.5 / 5
   Price    : Rs.899
...

Which one do you want to buy? (type the item/model name, you can just paste it loosely): Python in Action from croma

==================================================
HUMAN APPROVAL NEEDED
==================================================
Item       : Python in Action
Vendor     : Croma
Price      : Rs.950
Rating     : 4.6

Approve this purchase? (yes/no): yes

Your purchase of "Python in Action" from Croma is successful. (Order ID: order_TXxhD8CWDogj3L, Amount: Rs.950)
Remaining account balance: Rs.9050 (out of Rs.10000)

==================================================
SUMMARY - Final result
==================================================
Task               : buy me python handbook under 800
Budget Limit       : Rs.800.0
Selected Item      : Python in Action
Vendor             : Croma
Price              : Rs.950
User Approval      : True
Execution Result   :
   Item           : Python in Action
   Vendor         : Croma
   Amount         : Rs.950
   Razorpay Order ID : order_TXxhD8CWDogj3L
Total Spent So Far : Rs.950
Remaining Balance  : Rs.9050 (out of Rs.10000)
==================================================
```

## Design Decisions & Trade-offs

**Mock vendor catalog, LLM-generated.** No real marketplace API is used — Amazon's Product Advertising API and Flipkart's Affiliate API both require an approved affiliate account (sales history, days of approval time), and third-party aggregator APIs are paid and rate-limited. None of this was practical in a 2-day build. Instead of a hardcoded static list, the LLM generates plausible options on the fly for whatever the user asks for, so the agent isn't limited to a fixed set of product categories. **Extension point:** swap `search_options`'s body for a real affiliate/aggregator API call once credentials are available — the rest of the graph is unaffected, since everything downstream only depends on the `candidate_options` list shape.

**Payment reaches order creation, not capture.** The agent calls Razorpay's `order.create()` API — a real, verifiable call (visible in the Dashboard under Test Mode → Orders). It does not proceed to full payment *capture*, which would require either a manual checkout (typing in card details, which breaks the "agent acts autonomously" narrative this track rewards) or a tokenized saved-payment-method / mandate (Razorpay's Customer + Token API), which needs its own approval and setup flow — out of scope for 2 days. **Extension point:** in production, this final step would use a saved, tokenized payment method so the agent can complete the full charge without any human touching a card form — the same pattern Amazon's "1-click" uses.

**Human touchpoints kept to two.** Covered above — deliberately not more, to keep the agent's judgment (not the human's) doing the actual buying decisions.

**Persistent account balance via a local file, not full state.** LangGraph's checkpointer state resets its working values with each new `thread_id`; account balance is tracked separately in `account_balance.json` so it persists across runs regardless of session/thread boundaries.

**Scoped agent spending, modeled on RazorpayX Sub-Accounts.** The agent never has access to a user's full account balance — it operates against a fixed, allocated amount (`account_balance.json`, starting at ₹10,000) that only decreases as purchases happen. This is a deliberate safety pattern, not an incidental one: it mirrors [RazorpayX's Sub-Accounts feature](https://razorpay.com/docs/x/payouts/sub-accounts/), where a master bank account creates sub-accounts with their own static spending limits — a sub-account holder can only spend up to its allocated limit, never touching the master account directly. No AI agent should hold unscoped access to a user's full financial account; giving it a capped, dedicated "spending account" instead limits the blast radius of any mistake or misuse. In a production version, this local JSON file would be replaced by an actual RazorpayX Sub-Account, with the agent authenticated against that sub-account's own API keys — so the cap is enforced by Razorpay's infrastructure itself, not just application code.

## Failure Recovery — Real Bugs We Hit and Fixed

**Infinite retry loop.** The retry-counting logic for over-budget selections lived inside a conditional-edge *routing* function, not a graph *node*. LangGraph only persists a node's returned state — a routing function just reads state to pick the next step, so any mutation inside it is silently discarded. `retry_count` never actually incremented, so the loop never ended. Fix: moved the bookkeeping into an actual node.

**Budget wasn't enforced in search.** The LLM was only told "generate realistic options," with no budget constraint in the prompt — so a ₹800 budget could still return a ₹24,990 item. Fix: the budget is now stated explicitly in the prompt, twice, with a hard "must not exceed" instruction.

**No balance check before spending.** The agent could approve a purchase regardless of whether the wallet actually had enough left. Fix: `human_approval` now checks the remaining balance first and rejects outright ("insufficient balance") before any approval step runs.

**Balance reset every run.** Spend tracking lived only in LangGraph's per-session state, so it reset to the full amount on every new run instead of carrying over. Fix: balance now persists in a small external file (`account_balance.json`), read and updated independently of the session/thread state.

## Future Improvements

- Real marketplace API integration (Amazon/Flipkart affiliate, or a paid aggregator) in place of the LLM-generated mock catalog
- Tokenized saved payment methods for full autonomous payment capture, not just order creation
- Replacing `account_balance.json` with an actual RazorpayX Sub-Account, so spend limits are enforced by Razorpay's infrastructure rather than local application code
- Vector-store memory of past purchases to inform future comparisons (the `log_and_update_memory` node is a ready extension point)
- A lightweight Streamlit dashboard for a more visual approve/reject experience