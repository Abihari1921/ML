import streamlit as st

class BayesianNetwork:
    def __init__(self):
        self.probabilities = {}

    def add_probability(self, node, probability_table):
        self.probabilities[node] = probability_table

    def get_probability(self, node, evidence):
        table = self.probabilities[node]
        for condition, prob in table.items():
            if all(evidence.get(var) == val for var, val in condition.items()):
                return prob
        return 0

# Define the probabilities
# P(A)
P_A = {
    (): 0.2
}

# P(B|A)
P_B_given_A = {
    (('A', True),): 0.6,
    (('A', False),): 0.1
}

# P(C|B)
P_C_given_B = {
    (('B', True),): 0.7,
    (('B', False),): 0.2
}

# Initialize the network
bn = BayesianNetwork()
bn.add_probability('A', P_A)
bn.add_probability('B', P_B_given_A)
bn.add_probability('C', P_C_given_B)

def calculate_probability(evidence):
    # Example inference: P(C=True | evidence)
    A = evidence.get('A', None)

    if A is not None:
        # P(A=True)
        P_A_true = bn.get_probability('A', {})

        # P(B=True | A=True) and P(B=False | A=True)
        P_B_true_given_A_true = bn.get_probability('B', {'A': True})
        P_B_false_given_A_true = 1 - P_B_true_given_A_true

        # P(C=True | B=True) and P(C=True | B=False)
        P_C_true_given_B_true = bn.get_probability('C', {'B': True})
        P_C_true_given_B_false = bn.get_probability('C', {'B': False})

        # Calculate P(C=True | A=True)
        P_C_true_given_A_true = (P_C_true_given_B_true * P_B_true_given_A_true +
                                 P_C_true_given_B_false * P_B_false_given_A_true)
        return P_C_true_given_A_true
    else:
        return None

# Streamlit UI
st.title("Bayesian Network Inference")

st.write("### Set Evidence")
A_val = st.selectbox("A", [None, True, False], format_func=lambda x: "Select an option" if x is None else str(x))

evidence = {'A': A_val} if A_val is not None else {}

if st.button("Calculate P(C=True | evidence)"):
    result = calculate_probability(evidence)
    if result is not None:
        st.write(f"P(C=True | A={A_val}) = {result:.4f}")
    else:
        st.write("Please set evidence for A.")


