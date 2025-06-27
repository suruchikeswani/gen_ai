prompt = """
You are a helpful assistant skilled in converting natural language into SQL queries.

Task: Convert the following request into a valid SQL query.

Example:
Input: Show all employees in the Sales department hired after 2015.
Output: SELECT * FROM employees WHERE department = 'Sales' AND hire_date > '2015-01-01';

Now, convert this request:

Input: List the names of customers who have placed more than 3 orders.
Output:
"""