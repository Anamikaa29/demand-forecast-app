# 📊 Walmart-Style Product Demand Forecaster

This is a Streamlit-based web app that forecasts future product demand based on historical data, inspired by Walmart’s retail analytics.

## 🚀 Features
- Upload CSV with columns: `ds` (date), `y` (quantity), and `product` (item name)
- Interactive product selection
- Generates 30-day forecasts using Facebook Prophet
- Forecast plot with confidence intervals
- Downloadable forecast table

## 📂 Sample Input Format
| ds         | y    | product  |
|------------|------|----------|
| 2024-01-01 | 120  | T-Shirts |
| 2024-01-02 | 135  | T-Shirts |

## 📎 How to Use
1. Open the app (host locally or on Streamlit Cloud)
2. Upload your CSV file
3. Select a product from the dropdown
4. View the forecast graph and table

## ⚙️ Tech Stack
- Python
- Streamlit
- Facebook Prophet
- Pandas
- Plotly

## 👩‍💻 Built By
Anamikaa Sanjeev Nair  
MS in Technology Management @ UIUC

---

✨ Feel free to fork, contribute, or deploy it for your own inventory needs!
