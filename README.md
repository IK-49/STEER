# STEER: Statistical Twins for Educational Equity & Resources

**Carolina Data Challenge 2026** | Track: AI for Social Good / Social Sciences

STEER is an in-memory spatial retrieval engine that exposes and resolves **"The Composite Trap"**—where aggregate municipal hardship indices mask acute, localized civic crises across public schools.

By decomposing multi-domain community stress into standardized continuous vectors, STEER identifies empirical **statistical twin schools** across district and state borders, isolates acute crisis drivers via the **Composite Masking Index (CMI)**, simulates targeted relief via an algorithmic **Tipping Point Solver**, and deterministically synthesizes audit-ready statutory federal grant briefs in under 15 milliseconds.

---

## The Problem: The Composite Trap

State and federal education agencies distribute billions in targeted funding using aggregate 0–100 distress scores. This scalar compression creates false equivalencies between institutions facing diametrically opposed challenges:

| Dimension | Northeast Regional Biotech (Jamesville, NC) | D.H. Conley High (Greenville, NC) | Civic Divergence |
| :--- | :---: | :---: | :--- |
| **Composite Hardship Score** | **34** | **34** | **Identical Aggregate (0-point gap)** |
| **Safety / Crime Index** | **57** | **39** | **+18 pts Biotech** (Severe community safety crisis) |
| **Economic Vulnerability** | **46** | **34** | **+12 pts Biotech** (High economic isolation) |
| **Housing Instability** | **19** | **41** | **+22 pts Conley** (Severe student housing distress) |
| **Health Stress** | **28** | **36** | **+8 pts Conley** (Community health deficit) |

Under conventional funding formulas, both schools receive identical intervention packages. STEER decomposes these vectors to route capital where the crisis actually exists.

---

## Key Methodological Innovations

1. **Composite Masking Index (CMI):** Quantifies the degree to which an aggregate scalar obscures dimensional variance:
   $$\text{CMI} = \sqrt{\frac{1}{d} \sum_{j=1}^{d} (z_j - \bar{z})^2} = \sigma(\mathbf{z})$$
   A high CMI indicates severe dimensional imbalance, flagging institutions penalized by aggregate formula funding.
2. **Algorithmic Tipping Point Solver:** Uses a binary search solver to determine the exact minimum point reduction needed in the dominant crisis domain to normalize a school's CMI below $1.0\sigma$.
3. **Out-of-Sample Subspace $k$-NN:** Fits `StandardScaler` strictly across candidate pools, transforming queries without data leakage. Missing features are handled via dynamic subspace projection rather than global listwise deletion.
4. **Positive Deviance Retrieval:** Rather than only matching schools facing equivalent distress, STEER isolates operational mentors—institutions facing identical external community headwinds that have achieved superior educational attainment.
5. **Grounded LLM Grant Copilot:** Feeds verified mathematical outputs ($z$-scores, CMI, twin benchmarks, statutory targets) into a grounded prompt with a deterministic fallback, preventing hallucination while synthesizing an executive-level Statement of Need.

---

## Architecture & Visual Stack

* **Presentation Layer:** Streamlit with Plotly Graph Objects rendering 3-way stress radar topologies (`go.Scatterpolar`).
* **Analytical Engine:** In-memory execution via NumPy and Scikit-learn (`NearestNeighbors`, `StandardScaler`).
* **Inference Latency:** $<15\text{ ms}$ per end-to-end multi-domain query across 23,599 institutions with zero external API dependencies.

---

## Installation & Local Execution

Ensure Python 3.11+ is installed. From the project root:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install pinned dependencies
python -m pip install -r requirements.txt

# 3. Run validation test suite
python -m pytest

# 4. Launch STEER
streamlit run app.py
```