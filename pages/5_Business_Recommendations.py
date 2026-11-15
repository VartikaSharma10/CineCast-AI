"""
CineCast AI - Page 5: Smart Business Recommendations Engine
Deterministic, rule-based recommendation engine powered by actual ML model forecasts
and operational thresholds. Identifies screening interventions and portfolio strategies.
"""

import streamlit as st
import pandas as pd
from src.ui import init_page, render_page_header, render_kpi_card, render_synthetic_data_banner
from src.analytics import load_enriched_dataset
from src.recommendations import generate_screening_recommendations, generate_macro_insights

# Initialize page
init_page("Business Recommendations", icon="💡")

render_page_header(
    title="Smart Business Recommendations Engine",
    subtitle="Algorithmic operational directives and yield interventions driven by demand forecasts",
    badge_text="DECISION SUPPORT ENGINE"
)

render_synthetic_data_banner()

@st.cache_data(show_spinner="Evaluating operational rules against forecasts...")
def get_data():
    return load_enriched_dataset()

df_full = get_data()

# Generate recommendations dynamically from dataset and predictions
macro_insights = generate_macro_insights(df_full)
screening_actions = generate_screening_recommendations(df_full, max_items=50)

# ----------------- SECTION 1: MACRO STRATEGIC DIRECTIVES -----------------
st.markdown("### 🏛️ Executive Portfolio Directives")
st.markdown("Macro-level strategies synthesized across 6,500+ screening records and historical booking cycles.")

macro_col1, macro_col2, macro_col3 = st.columns(3)
for col, insight in zip([macro_col1, macro_col2, macro_col3], macro_insights):
    with col:
        st.markdown(
            f"""
            <div class="cinema-card" style="border-top: 3px solid {insight['color']}; min-height: 220px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                        <span class="badge-sage">{insight['badge']}</span>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #172321; margin-bottom: 0.6rem;">
                        {insight['title']}
                    </div>
                    <div style="font-size: 0.88rem; color: #2B3D38; line-height: 1.5; font-weight: 500;">
                        {insight['description']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- SECTION 2: SCREENING ACTION QUEUE -----------------
st.markdown("### ⚡ Screening Operational Action Queue")
st.markdown("Individual scheduled screenings requiring operational intervention based on ML demand predictions.")

# Filters for recommendations
filter_c1, filter_c2 = st.columns([1, 1])

all_rule_types = ["All Action Types"] + sorted(list({r["rule_type"] for r in screening_actions}))
all_priorities = ["All Priorities"] + ["Urgent", "High", "Opportunity", "Medium"]

with filter_c1:
    selected_rule = st.selectbox("Filter by Action Type", options=all_rule_types)

with filter_c2:
    selected_priority = st.selectbox("Filter by Priority", options=all_priorities)

filtered_recs = screening_actions
if selected_rule != "All Action Types":
    filtered_recs = [r for r in filtered_recs if r["rule_type"] == selected_rule]
if selected_priority != "All Priorities":
    filtered_recs = [r for r in filtered_recs if r["priority"] == selected_priority]

# Summary counters
total_actions = len(filtered_recs)
urgent_count = sum(1 for r in filtered_recs if r["priority"] == "Urgent")
high_count = sum(1 for r in filtered_recs if r["priority"] == "High")
opp_count = sum(1 for r in filtered_recs if r["priority"] == "Opportunity")

qc1, qc2, qc3, qc4 = st.columns(4)
with qc1:
    st.markdown(render_kpi_card("Pending Action Items", f"{total_actions}", "Filtered screenings queue"), unsafe_allow_html=True)
with qc2:
    st.markdown(render_kpi_card("Urgent Interventions", f"{urgent_count}", "Capacity > 90% alert"), unsafe_allow_html=True)
with qc3:
    st.markdown(render_kpi_card("High Priority Flags", f"{high_count}", "Capacity or velocity alert"), unsafe_allow_html=True)
with qc4:
    st.markdown(render_kpi_card("Surge Yield Opportunities", f"{opp_count}", "Dynamic pricing opportunities", is_gold=True), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if not filtered_recs:
    st.info("No screenings in the queue match the chosen filter combination.")
else:
    for rec in filtered_recs[:15]:
        prio = rec["priority"]
        badge_style = "badge-urgent" if prio == "Urgent" else ("badge-gold" if prio == "Opportunity" else "badge-forest")
        
        st.markdown(
            f"""
            <div class="cinema-card" style="margin-bottom: 1rem; border-left: 4px solid {'#8A1F1F' if prio=='Urgent' else ('#8C6B2B' if prio=='Opportunity' else '#172321')};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; margin-bottom: 0.5rem;">
                    <div>
                        <span style="font-weight: 700; font-size: 1.1rem; color: #172321;">{rec['movie_title']}</span>
                        <span style="color: #384A45; font-size: 0.85rem; margin-left: 8px; font-weight: 500;">— {rec['theatre_name']} | {rec['show_date']} at {rec['show_time']}</span>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <span class="{badge_style}">{prio.upper()}</span>
                        <span class="badge-sage">{rec['predicted_occupancy']} OCCUPANCY</span>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #2B3D38; margin-bottom: 0.5rem;">
                    <strong>Trigger Condition:</strong> {rec['metric_trigger']}
                </div>
                <div style="background-color: #F0ECE2; border: 1px solid #DFD9CB; border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0;">
                    <div style="font-size: 0.92rem; font-weight: 600; color: #172321; margin-bottom: 0.3rem;">
                        📋 Recommended Action: {rec['action']}
                    </div>
                    <div style="font-size: 0.85rem; color: #1A4B3C; font-weight: 600;">
                        💡 Projected Impact: {rec['projected_impact']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Export Action Plan CSV
st.markdown("<br>", unsafe_allow_html=True)
if filtered_recs:
    rec_df = pd.DataFrame(filtered_recs)
    csv_bytes = rec_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Action Queue to CSV",
        data=csv_bytes,
        file_name="cinecast_operational_recommendations.csv",
        mime="text/csv",
        use_container_width=False
    )
