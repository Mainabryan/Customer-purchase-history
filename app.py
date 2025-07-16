

import streamlit as st

class Calculator:
    def __init__(self):
        self.operations = {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "/": lambda x, y: "Cannot divide by zero." if y == 0 else x / y
        }

    def add_operation(self, symbol, function):
        """Dynamically add new operations to the calculator"""
        self.operations[symbol] = function

    def calculate(self, num1, operation, num2):
        """Performs the calculation based on the chosen operation"""
        if operation not in self.operations:
            return f"Error: '{operation}' is not a valid operation."
        func = self.operations[operation]
        return func(num1, num2)

# Streamlit app
def main():
    st.title("Simple Calculator")

    # Create calculator instance
    calc = Calculator()

    # Input fields
    num1 = st.number_input("Enter first number", format="%f")
    operation = st.selectbox("Select operation", list(calc.operations.keys()))
    num2 = st.number_input("Enter second number", format="%f")

    # Calculate button
    if st.button("Calculate"):
        try:
            result = calc.calculate(num1, operation, num2)
            if isinstance(result, str):
                st.error(result)
            else:
                st.success(f"Result: {result}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
