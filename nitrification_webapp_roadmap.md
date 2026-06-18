# Nitrification Prediction Web App — Development Roadmap

## 📋 สรุปเป้าหมาย
สร้าง Web Application ที่ให้ user input ค่า environment parameters แล้วระบบจะ **simulate nitrification kinetics** โดยใช้ Monod + Inhibition Models

---

## 🏗️ ขั้นตอนการพัฒนา (4 ขั้นหลัก)

### Phase 1: Backend — สูตรคณิตศาสตร์ → Code

**สิ่งที่ต้องทำ:**
- [ ] สร้างโมเดล ODE จากสูตรในโน้ต
  - Monod kinetics: `μ = μ_max * S / (Ks + S)`
  - Nitrification: `dNH4/dt = -(μ*X/Y) * frac`
  - Inhibition models (Competitive, Uncompetitive, Non-competitive)
  
- [ ] เขียน Python functions
  ```python
  def nitrification_model(t, y, params):
      """ODE model สำหรับ nitrification"""
      S, X = y  # Substrate (NH4), Biomass
      μ = monod(S, params['mu_max'], params['Ks'])
      # Add inhibition if needed
      dS_dt = -(μ * X / params['Y'])
      dX_dt = μ * X
      return [dS_dt, dX_dt]
  
  def solve_nitrification(init_cond, params, t_span):
      """รัน ODE solver"""
      from scipy.integrate import solve_ivp
      sol = solve_ivp(nitrification_model, t_span, init_cond, 
                      args=(params,), dense_output=True)
      return sol
  ```

- [ ] Test กับค่า default parameters
  - μ_max = 0.5-1.5 /day
  - Ks = 2-10 mg/L
  - Y = 0.1-0.3
  - KI = 50-100 mg/L (ถ้ามี inhibitor)

---

### Phase 2: API — เชื่อม Backend กับ Frontend

**ทีมพัฒนา Framework:**
- [ ] เลือก FastAPI (รวดเร็ว + async)
  ```python
  from fastapi import FastAPI
  app = FastAPI()
  
  @app.post("/predict")
  async def predict(params: dict):
      """
      Input: {
          "NH4_init": 50,
          "X_init": 0.1,
          "time_days": 7,
          "mu_max": 1.0,
          "Ks": 5,
          "Y": 0.2,
          "inhibition_type": "none",  # or "competitive", "uncompetitive"
          "I": 0,
          "KI": 50
      }
      Output: {
          "time": [...],
          "NH4": [...],
          "X": [...]
      }
      """
      sol = solve_nitrification(...)
      return {"success": True, "data": {...}}
  ```

- [ ] Error handling (ถ้า user input ค่าแปลก)

---

### Phase 3: Frontend — User Interface

**เลือกวิธีที่เหมาะ:**

**ตัวเลือก A: Streamlit (แนะนำสำหรับเริ่มต้น)**
```python
import streamlit as st
import plotly.graph_objects as go

st.title("Nitrification Prediction")

col1, col2 = st.columns(2)
with col1:
    NH4_init = st.slider("Initial NH4 (mg/L)", 1, 200, 50)
    mu_max = st.slider("μ_max (/day)", 0.1, 2.0, 1.0)
    
with col2:
    Ks = st.slider("Ks (mg/L)", 1, 20, 5)
    Y = st.slider("Y (yield)", 0.05, 0.5, 0.2)

if st.button("Simulate"):
    sol = solve_nitrification(...)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name="NH4"))
    st.plotly_chart(fig)
```

**ตัวเลือก B: React + Plotly (ถ้าอยากเหมือน professional product)**
- สร้างฟอร์มด้วย React components
- Axios ส่ง request ไป FastAPI
- Plotly.js plot กราฟบน frontend

---

### Phase 4: (Optional) ML Layer — Accelerate Prediction

**เฉพาะถ้าต้องการ prediction เร็วมาก:**
- [ ] เก็บผล simulate เป็น dataset
- [ ] Train ML model (Neural Network / Random Forest)
  - Input: [NH4, Ks, mu_max, Y, I]
  - Output: [final_NH4, time_to_complete, ...]
- [ ] Replace ODE solver ด้วย ML inference (ถ้า predict ต้องใช้ลัดขั้น)

---

## 📁 File Structure

```
nitrification-webapp/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── models/
│   │   ├── nitrification.py   # ODE model
│   │   └── inhibition.py      # Inhibition logic
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── app.py                 # Streamlit (if chosen)
│   ├── requirements.txt
│   └── components/            # (if React)
│
├── notebooks/
│   └── validation.ipynb       # Test model กับ literature values
│
└── README.md
```

---

## 🔧 Tech Stack (แนะนำ)

| ส่วน | เครื่องมือ | ทำไม |
|------|-----------|------|
| Backend | FastAPI | รวดเร็ว, async, OpenAPI docs |
| ODE Solver | scipy.integrate | industry standard |
| Frontend | Streamlit | ขึ้น Python ได้เลย, ไม่ต้อง HTML/CSS |
| Visualization | Plotly | interactive graphs |
| Database (optional) | PostgreSQL + SQLAlchemy | บันทึก simulation history |
| Deploy | Railway / Render | free tier ใช้ได้ |

---

## 📊 Parameter Reference (จากโน้ต)

| Parameter | Unit | Range | Default | Note |
|-----------|------|-------|---------|------|
| μ_max | /day | 0.5-1.5 | 1.0 | AOA (Ammonia oxidizer) |
| Ks | mg/L | 1-10 | 2 | Affinity untuk NH4 |
| Y | - | 0.1-0.3 | 0.15 | Yield coefficient |
| KI | mg/L | 50-200 | 100 | Inhibition constant |
| Temperature | °C | 5-40 | 20 | Affects μ_max (optional) |

---

## ✅ Testing Checklist

- [ ] Unit test: ODE solver ให้ผลถูกต้อง
- [ ] Integration test: API endpoint ทำงาน
- [ ] UI test: Slider/input ส่งค่าได้
- [ ] Validation: ผล simulate match กับ literature
- [ ] Edge cases: ค่า extreme input (NH4=1000, KI=0)

---

## 🚀 Phase Release Plan

1. **MVP (Week 1-2):** Streamlit + backend → user ลองใช้ได้
2. **v1.0 (Week 3-4):** FastAPI + validation + docs
3. **v2.0 (Future):** React frontend + ML acceleration
4. **v3.0 (Future):** Add more models (denitrification, etc.)

---

## 📚 Resources

- Monod kinetics: Biology/Bioprocess textbooks
- Inhibition models: Paper by Haldane (1930) / Lineweaver-Burk
- scipy.integrate docs: https://docs.scipy.org/doc/scipy/reference/integrate.html
- FastAPI tutorial: https://fastapi.tiangolo.com/
- Streamlit docs: https://docs.streamlit.io/
