import streamlit as st
import pandas as pd
import numpy as np

# Criteria weights input
st.title("Plagiarism Detection Strategy Selection using TOPSIS")

st.sidebar.header("Criteria Weights")
weight_ghostwriting = st.sidebar.slider("Ghostwriting (C1)", 0.0, 1.0, 0.2)
weight_syntax = st.sidebar.slider("Syntax Preservation (C2)", 0.0, 1.0, 0.2)
weight_character = st.sidebar.slider("Character Preservation (C3)", 0.0, 1.0, 0.2)
weight_semantic = st.sidebar.slider("Semantic Preservation (C4)", 0.0, 1.0, 0.2)
weight_concept = st.sidebar.slider("Concept Preservation (C5)", 0.0, 1.0, 0.2)

# Ensure the sum of weights is 1
total_weight = weight_ghostwriting + weight_syntax + weight_character + weight_semantic + weight_concept
if total_weight != 1.0:
    st.sidebar.error("The sum of all weights must be 1.0")

# Define the criteria weights
weights = np.array([
    weight_ghostwriting,
    weight_syntax,
    weight_character,
    weight_semantic,
    weight_concept
])

# Alternatives and criteria data
data = {
    'Alternative': ['Machine Learning', 'NLP with AI', 'TF-IDF', 'N-gram', 'LSA', 'Stylometry'],
    'C1': [0.8, 0.9, 0.5, 0.6, 0.7, 0.4],
    'C2': [0.9, 0.8, 0.6, 0.5, 0.7, 0.6],
    'C3': [0.7, 0.6, 0.9, 0.8, 0.5, 0.4],
    'C4': [0.6, 0.7, 0.5, 0.9, 0.8, 0.6],
    'C5': [0.5, 0.4, 0.8, 0.7, 0.6, 0.9]
}

df = pd.DataFrame(data)

# Normalize the decision matrix
normalized_df = df.iloc[:, 1:].div(np.sqrt((df.iloc[:, 1:] ** 2).sum(axis=0)), axis=1)

# Calculate the weighted normalized decision matrix
weighted_normalized_df = normalized_df * weights

# Determine FPIS and FNIS
FPIS = weighted_normalized_df.max()
FNIS = weighted_normalized_df.min()

# Calculate the distance of each alternative from FPIS and FNIS
distance_to_FPIS = np.sqrt(((weighted_normalized_df - FPIS) ** 2).sum(axis=1))
distance_to_FNIS = np.sqrt(((weighted_normalized_df - FNIS) ** 2).sum(axis=1))

# Calculate the relative closeness to the ideal solution
relative_closeness = distance_to_FNIS / (distance_to_FPIS + distance_to_FNIS)

# Rank the alternatives
df['Relative Closeness'] = relative_closeness
df['Rank'] = df['Relative Closeness'].rank(ascending=False)

# Display the raw data and the results
st.write("### Criteria Weights")
st.write(f"C1 (Ghostwriting): {weight_ghostwriting}")
st.write(f"C2 (Syntax Preservation): {weight_syntax}")
st.write(f"C3 (Character Preservation): {weight_character}")
st.write(f"C4 (Semantic Preservation): {weight_semantic}")
st.write(f"C5 (Concept Preservation): {weight_concept}")

st.write("### Alternatives and Criteria Scores")
st.dataframe(df)

st.write("### Results")
st.dataframe(df[['Alternative', 'Relative Closeness', 'Rank']])

# Plot the data
st.write("### Graph of Relative Closeness vs Rank")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(df['Rank'], df['Relative Closeness'], marker='o', linestyle='-', color='b')
ax.set_xlabel('Rank')
ax.set_ylabel('Relative Closeness')
ax.set_title('Relative Closeness vs Rank')

# Display the graph
st.pyplot(fig)


