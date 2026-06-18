# SKILL: Building Mechanistic Model Web Applications

> Guidance for creating web applications that combine **mathematical models** (ODE-based), **backend APIs**, and **interactive frontends** to visualize and predict biochemical/environmental engineering processes.

---

## When to Use This Skill

Use this skill when:
- You have a **mechanistic mathematical model** (ODE, PDE, stoichiometric equations) that needs to be interactive
- You want to let **users input parameters** and see simulation results in real-time
- You're building a tool for **teaching, research, or prediction** of environmental/bioprocess systems
- You need to wrap **numerical solvers** (scipy, numpy) with a web interface
- You don't have experimental data yet but want to **simulate from first principles**

**Examples:**
- Nitrification/denitrification kinetics
- Bioreactor batch/fed-batch simulations
- Chemical equilibrium calculators
- Adsorption isotherms
- Biodegradation rate models

---

## Architecture Overview

```
┌─────────────┐         ┌─────────────┐         ┌──────────────┐
│   Frontend  │────────▶│  Backend    │────────▶│   Solver     │
│ (Streamlit/ │ HTTP    │  (FastAPI)  │ Python  │  (scipy ODE) │
│   React)    │◀────────│             │◀────────│              │
└─────────────┘ JSON    └─────────────┘ Results └──────────────┘
   User Input               Routing              Math/Physics
```

**Data Flow:**
1. User adjusts slider/input → Frontend
2. Frontend sends JSON payload → Backend API
3. Backend calls ODE solver with params → Numeric computation
4. Results returned as JSON → Frontend plots

---

## Core Components

### 1. Mathematical Model Definition

**Step 1.1: Translate equations to Python**

Given a system of ODEs:
$$\frac{dS}{dt} = -\mu(S) \cdot X$$
$$\frac{dX}{dt} = Y \cdot \mu(S) \cdot X$$

Implement as:
```python
def kinetic_model(t, y, params):
    """
    y = [S, X] where S=substrate, X=biomass
    params = dict with {'mu_max': ..., 'Ks': ..., 'Y': ...}
    returns dy/dt
    """
    S, X = y
    mu_max = params['mu_max']
    Ks = params['Ks']
    Y = params['Y']
    
    # Monod kinetics
    mu = mu_max * (S / (Ks + S))
    
    # ODEs
    dS_dt = -mu * X
    dX_dt = Y * mu * X
    
    return [dS_dt, dX_dt]
```

**Step 1.2: Add inhibition (if needed)**

```python
def inhibition_factor(I, KI, inhibition_type='none'):
    """Compute (1 + I/KI) term or variants"""
    if inhibition_type == 'none':
        return 1.0
    elif inhibition_type == 'competitive':
        return 1 + (I / KI)
    elif inhibition_type == 'uncompetitive':
        return 1 + (I / KI)
    elif inhibition_type == 'non-competitive':
        return 1 + (I / KI)
    return 1.0

def kinetic_model_with_inhibition(t, y, params):
    S, X = y
    mu_max = params['mu_max']
    Ks = params['Ks']
    Y = params['Y']
    I = params.get('inhibitor', 0)
    KI = params.get('KI', 100)
    inh_type = params.get('inhibition_type', 'none')
    
    inh_factor = inhibition_factor(I, KI, inh_type)
    
    if inh_type == 'competitive':
        mu = mu_max * (S / (Ks * inh_factor + S))
    elif inh_type == 'uncompetitive':
        mu = mu_max * (S / (inh_factor * (Ks + S)))
    elif inh_type == 'non-competitive':
        mu = mu_max * (S / (Ks + S * inh_factor))
    else:
        mu = mu_max * (S / (Ks + S))
    
    dS_dt = -mu * X
    dX_dt = Y * mu * X
    return [dS_dt, dX_dt]
```

**Step 1.3: Validation — Check against literature**

```python
# Test with known parameters from literature
test_params = {
    'mu_max': 0.8,  # /day
    'Ks': 2.0,      # mg/L
    'Y': 0.15,
}

y0 = [50, 0.1]  # [NH4_init, X_init]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 100)

sol = solve_ivp(kinetic_model, t_span, y0, 
                args=(test_params,), t_eval=t_eval)

# Plot and compare with reference data
plt.plot(sol.t, sol.y[0], label='Simulated NH4')
plt.xlabel('Time (days)')
plt.ylabel('Concentration (mg/L)')
plt.show()
```

---

### 2. Backend API

**Step 2.1: Create FastAPI endpoint**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import json
from scipy.integrate import solve_ivp

app = FastAPI(title="Nitrification Predictor")

class SimulationRequest(BaseModel):
    """Request payload schema"""
    S_init: float = 50          # Initial substrate (mg/L)
    X_init: float = 0.1         # Initial biomass
    time_days: float = 10
    mu_max: float = 0.8
    Ks: float = 2.0
    Y: float = 0.15
    inhibition_type: str = "none"
    inhibitor: float = 0
    KI: float = 100

class SimulationResponse(BaseModel):
    """Response payload schema"""
    success: bool
    message: str
    data: dict = None
    error: str = None

@app.post("/predict", response_model=SimulationResponse)
async def predict(req: SimulationRequest):
    """
    Run nitrification simulation with given parameters.
    Returns time series of substrate and biomass.
    """
    try:
        params = {
            'mu_max': req.mu_max,
            'Ks': req.Ks,
            'Y': req.Y,
            'inhibitor': req.inhibitor,
            'KI': req.KI,
            'inhibition_type': req.inhibition_type,
        }
        
        y0 = [req.S_init, req.X_init]
        t_span = (0, req.time_days)
        t_eval = np.linspace(0, req.time_days, 200)
        
        sol = solve_ivp(kinetic_model_with_inhibition, t_span, y0,
                        args=(params,), t_eval=t_eval, method='RK45')
        
        if not sol.success:
            return SimulationResponse(
                success=False,
                message="ODE solver failed",
                error=sol.message
            )
        
        return SimulationResponse(
            success=True,
            message="Simulation completed",
            data={
                'time': sol.t.tolist(),
                'S': sol.y[0].tolist(),
                'X': sol.y[1].tolist(),
                'params': params
            }
        )
    except Exception as e:
        return SimulationResponse(
            success=False,
            message="Error",
            error=str(e)
        )

@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Step 2.2: Error handling & validation**

```python
from fastapi import HTTPException

@app.post("/predict")
async def predict(req: SimulationRequest):
    # Validate input ranges
    if req.mu_max <= 0 or req.mu_max > 5:
        raise HTTPException(status_code=400, 
                          detail="mu_max must be between 0 and 5")
    if req.time_days <= 0 or req.time_days > 365:
        raise HTTPException(status_code=400,
                          detail="time_days must be between 0 and 365")
    # ... rest of function
```

---

### 3. Frontend (Streamlit - Recommended for MVP)

**Step 3.1: Basic structure**

```python
# app.py
import streamlit as st
import plotly.graph_objects as go
import requests
import json

st.set_page_config(page_title="Nitrification Predictor", layout="wide")
st.title("🧬 Nitrification Kinetics Simulator")

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

col1, col2 = st.sidebar.columns(2)
with col1:
    S_init = st.sidebar.slider("NH₄⁺ Init (mg/L)", 1, 200, 50)
    mu_max = st.sidebar.slider("μ_max (/day)", 0.1, 2.0, 0.8, step=0.1)
with col2:
    X_init = st.sidebar.slider("Biomass (g/L)", 0.01, 1.0, 0.1, step=0.01)
    Ks = st.sidebar.slider("K_s (mg/L)", 0.5, 20.0, 2.0, step=0.5)

Y = st.sidebar.slider("Y (yield)", 0.05, 0.5, 0.15, step=0.01)
time_days = st.sidebar.slider("Simulation time (days)", 1, 365, 10)

# Inhibition options
st.sidebar.subheader("Inhibition")
inh_type = st.sidebar.selectbox("Type", 
    ["none", "competitive", "uncompetitive", "non-competitive"])
if inh_type != "none":
    inhibitor = st.sidebar.slider("Inhibitor (mg/L)", 0, 500, 0)
    KI = st.sidebar.slider("K_I (mg/L)", 1, 500, 100)
else:
    inhibitor = 0
    KI = 100

# Run simulation
if st.sidebar.button("▶ Simulate"):
    payload = {
        'S_init': S_init,
        'X_init': X_init,
        'time_days': time_days,
        'mu_max': mu_max,
        'Ks': Ks,
        'Y': Y,
        'inhibition_type': inh_type,
        'inhibitor': inhibitor,
        'KI': KI,
    }
    
    with st.spinner("Running solver..."):
        try:
            # Call FastAPI endpoint
            response = requests.post(
                "http://localhost:8000/predict",
                json=payload
            )
            result = response.json()
            
            if result['success']:
                data = result['data']
                
                # Plot results
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['time'], y=data['S'],
                    name="NH₄⁺", mode='lines'
                ))
                fig.add_trace(go.Scatter(
                    x=data['time'], y=data['X'],
                    name="Biomass", mode='lines', yaxis='y2'
                ))
                fig.update_layout(
                    title="Nitrification Dynamics",
                    xaxis_title="Time (days)",
                    yaxis_title="NH₄⁺ (mg/L)",
                    yaxis2=dict(title="Biomass (g/L)", overlaying='y', side='right')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final NH₄⁺", f"{data['S'][-1]:.2f} mg/L")
                with col2:
                    st.metric("Final Biomass", f"{data['X'][-1]:.4f} g/L")
                with col3:
                    st.metric("Removal %", 
                             f"{(1 - data['S'][-1]/S_init)*100:.1f}%")
            else:
                st.error(f"Simulation failed: {result['error']}")
        
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
```

**Step 3.2: Add documentation**

```python
with st.expander("📖 Model Documentation"):
    st.markdown("""
    ### Monod Kinetics
    $$\\mu = \\mu_{max} \\cdot \\frac{S}{K_s + S}$$
    
    ### Nitrification ODEs
    $$\\frac{dNH_4}{dt} = -\\frac{\\mu \\cdot X}{Y}$$
    $$\\frac{dX}{dt} = \\mu \\cdot X$$
    
    ### Parameters
    - **μ_max**: Maximum specific growth rate (/day)
    - **K_s**: Half-saturation constant (mg/L)
    - **Y**: Yield coefficient (-)
    - **K_I**: Inhibition constant (mg/L)
    """)
```

---

### 4. Frontend (React - Optional, for Production)

```javascript
// components/SimulationForm.js
import React, { useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import axios from 'axios';

const SimulationForm = () => {
  const [params, setParams] = useState({
    S_init: 50,
    mu_max: 0.8,
    // ... other params
  });
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleSimulate = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/predict', params);
      setResult(response.data);
      plotResults(response.data);
    } catch (error) {
      console.error('Simulation failed:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const plotResults = (data) => {
    const trace = {
      x: data.data.time,
      y: data.data.S,
      type: 'scatter',
      name: 'NH₄⁺'
    };
    Plotly.newPlot('plot', [trace]);
  };
  
  return (
    <div>
      <input 
        type="number" 
        value={params.S_init}
        onChange={(e) => setParams({...params, S_init: parseFloat(e.target.value)})}
      />
      <button onClick={handleSimulate} disabled={loading}>
        {loading ? 'Simulating...' : 'Simulate'}
      </button>
      <div id="plot"></div>
    </div>
  );
};

export default SimulationForm;
```

---

## Common Patterns & Best Practices

### Parameter Normalization
```python
def normalize_params(params):
    """Ensure params are within realistic ranges"""
    params['mu_max'] = max(0.01, min(5.0, params['mu_max']))
    params['Ks'] = max(0.1, min(100, params['Ks']))
    params['Y'] = max(0.01, min(1.0, params['Y']))
    return params
```

### Caching Results (for repeated queries)
```python
from functools import lru_cache
import json

@lru_cache(maxsize=128)
def cached_simulation(param_hash):
    """Cache results keyed by parameter hash"""
    return solve_ivp(...)

def get_cache_key(params):
    return json.dumps(params, sort_keys=True)
```

### Sensitivity Analysis
```python
def sensitivity_analysis(base_params, param_name, range_pct=20):
    """Vary one parameter and see effect"""
    results = {}
    base_val = base_params[param_name]
    
    for multiplier in np.linspace(0.8, 1.2, 10):
        test_params = base_params.copy()
        test_params[param_name] = base_val * multiplier
        sol = solve_ivp(kinetic_model, ...)
        results[multiplier] = sol.y[-1]  # Final state
    
    return results
```

---

## Deployment Checklist

- [ ] Test all edge cases (extreme parameter values)
- [ ] Add unit tests for ODE model
- [ ] Add integration tests for API
- [ ] Document parameter ranges and defaults
- [ ] Set up error logging
- [ ] Add rate limiting (if public API)
- [ ] Choose hosting (Railway, Render, AWS)
- [ ] Create README with examples

**Deploy Streamlit:**
```bash
# requirements.txt
streamlit==1.28.0
requests==2.31.0
plotly==5.17.0

# Run locally
streamlit run app.py

# Deploy to Streamlit Cloud
git push to GitHub → connect to Streamlit Cloud
```

**Deploy FastAPI:**
```bash
# requirements.txt
fastapi==0.104.0
uvicorn==0.24.0
scipy==1.11.4
numpy==1.24.3

# Run locally
uvicorn main:app --reload

# Deploy to Railway/Render
railway up  # or git push to Render
```

---

## Gotchas & Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| ODE solver diverges | Stiff system or bad parameters | Use `method='BDF'` or tune `max_step` |
| Slow simulation | Too many timesteps or fine grid | Reduce `t_eval` points or raise `rtol` |
| Frontend doesn't update | CORS issue | Add CORS middleware to FastAPI |
| User inputs produce NaN | Division by zero (Ks=0) | Validate inputs before passing to solver |
| Negative concentrations | Numerical artifacts | Clamp results to [0, inf) |

---

## References & Resources

- **ODE Solvers:** scipy.integrate.solve_ivp [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
- **FastAPI:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **Streamlit:** [docs.streamlit.io](https://docs.streamlit.io/)
- **Monod Kinetics:** Monod, J. (1942) — foundational paper
- **Inhibition Models:** Lineweaver & Burk (1934), or modern reviews
- **Nitrification:** Daigger & Littleton (2014) — activated sludge textbook

---

## Next Steps After MVP

1. **Add more processes**: Denitrification, nitrite oxidation
2. **Temperature correction**: Adjust μ_max with Arrhenius equation
3. **Multiple inhibitors**: Combine multiple I terms
4. **Export results**: CSV, JSON, PDF reports
5. **Optimization**: Find parameters that minimize/maximize something
6. **Machine learning**: Train surrogate model for faster inference
