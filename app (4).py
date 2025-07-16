```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import uuid
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Set page configuration for wide layout and custom theme
st.set_page_config(
    page_title="Market Basket Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #FFFFFF; padding: 20px;}
    .stSidebar {background-color: #F8F9FA;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stSlider label {font-weight: bold;}
    .stExpander {background-color: #F1F3F5; border-radius: 5px;}
    .css-1d391kg {padding: 1rem;}
    h1 {color: #2C3E50; font-family: 'Arial', sans-serif;}
    h2, h3 {color: #34495E; font-family: 'Arial', sans-serif;}
    .stDataFrame {font-size: 14px;}
    .footer {text-align: center; color: #7F8C8D; margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🛒 Market Basket Analysis - Supermarket Insights")

# Load and preprocess data (assuming the same dataset as provided)
@st.cache_data
def load_data():
    data = pd.read_csv('Market_Basket_Optimisation.csv')
    cleaned_transactions = []
    for _, row in data.iterrows():
        items = [str(item).strip().lower() for item in row if pd.notna(item)]
        items = list(set(items))
        if len(items) > 1:
            cleaned_transactions.append(items)
    
    te = TransactionEncoder()
    te_array = te.fit_transform(cleaned_transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    return rules

rules = load_data()

# Sidebar
with st.sidebar:
    st.header("Filter Options")
    
    # Sliders for filtering
    min_support = st.slider("Minimum Support", min_value=0.01, max_value=0.1, value=0.01, step=0.005)
    min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
    min_lift = st.slider("Minimum Lift", min_value=1.0, max_value=3.0, value=1.0, step=0.1)
    
    # Dropdown for specific item filtering
    all_items = set()
    for ante, cons in zip(rules['antecedents'], rules['consequents']):
        all_items.update(ante)
        all_items.update(cons)
    all_items = sorted(list(all_items))
    selected_item = st.selectbox("Filter by Item (Optional)", ["All"] + all_items)
    
    # Sort option
    sort_by = st.selectbox("Sort Rules By", ["Lift", "Confidence"])

# Filter rules based on sidebar inputs
filtered_rules = rules[
    (rules['support'] >= min_support) &
    (rules['confidence'] >= min_confidence) &
    (rules['lift'] >= min_lift)
]

# Apply item filter if selected
if selected_item != "All":
    filtered_rules = filtered_rules[
        filtered_rules['antecedents'].apply(lambda x: selected_item in x) |
        filtered_rules['consequents'].apply(lambda x: selected_item in x)
    ]

# Sort rules
filtered_rules = filtered_rules.sort_values(by=sort_by.lower(), ascending=False)

# Summary Insights
st.subheader("Summary Insights")
insights = [
    f"**Top Rule by Lift**: {filtered_rules.iloc[0]['antecedents']} → {filtered_rules.iloc[0]['consequents']} (Lift: {filtered_rules.iloc[0]['lift']:.2f})",
    f"**Total Rules Found**: {len(filtered_rules)} rules after applying filters",
    f"**Most Frequent Item**: {'mineral water' if 'mineral water' in all_items else all_items[0]} appears in many strong associations"
]
st.info("\n".join(insights))

# Main Panel - Rules Table
st.subheader("Association Rules")
st.write(f"Total Rules Displayed: {len(filtered_rules)}")
st.dataframe(
    filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].style.format({
        'support': '{:.3f}',
        'confidence': '{:.3f}',
        'lift': '{:.3f}'
    })
)

# Download button for filtered rules
csv = filtered_rules.to_csv(index=False)
st.download_button(
    label="Download Rules as CSV",
    data=csv,
    file_name="market_basket_rules.csv",
    mime="text/csv"
)

# Explanations
with st.expander("Understanding the Metrics"):
    st.markdown("""
    - **Support**: The percentage of transactions containing a particular combination of items.
    - **Confidence**: The probability that a transaction containing the antecedent also contains the consequent.
    - **Lift**: The ratio of observed support to expected support if the items were independent. Lift > 1 indicates a strong association.
    """)

# Visualizations
st.subheader("Visualizations")

# Bar Chart: Top 10 Rules by Lift
top_10_rules = filtered_rules.head(10).copy()
top_10_rules['rule'] = top_10_rules.apply(lambda x: f"{x['antecedents']} → {x['consequents']}", axis=1)
fig_bar = px.bar(
    top_10_rules,
    x='lift',
    y='rule',
    title="Top 10 Association Rules by Lift",
    labels={'lift': 'Lift', 'rule': 'Rule'},
    color='confidence',
    color_continuous_scale='Viridis'
)
fig_bar.update_layout(showlegend=True, margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig_bar, use_container_width=True)

# Network Graph
st.subheader("Network Graph of Associations")
G = nx.DiGraph()
for idx, row in filtered_rules.head(20).iterrows():
    for ante in row['antecedents']:
        for cons in row['consequents']:
            G.add_edge(ante, cons, weight=row['lift'])

pos = nx.spring_layout(G)
edge_x = []
edge_y = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition='top center',
    hoverinfo='text',
    marker=dict(size=10, color='#4CAF50')
)

fig_network = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(show