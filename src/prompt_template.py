prompt = """Below are some details of a bank transaction. Your task is to classify the transaction into one of the 
following 'categories':
categories:
- Charity
- Fuel
- Cash Withdrawal
- Groceries
- Miscellaneous
- Utilities
- Salary
- Shopping
- Investment
- Healthcare
- Insurance
- Dining
- Entertainment
- Travel
- Education
- Taxes
You will be provided with a 'Description' of the transaction below and a 'Merchant_Name' 
which denotes the party involved in the transaction within triple backticks below.
Reply only with the output Category as a JSON with the below field:
Category:
Do not add any additional data
```
Description: {desc}
Merchant_Name: {merch}
```
"""