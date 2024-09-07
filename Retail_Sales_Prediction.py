# Importing required packages:
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# Function to load the pre-trained model
def load_model(model_path):
        with open(model_path, 'rb') as file:  # Open the model file in read-binary mode
            model = pickle.load(file)  # Load the model using pickle
        return model  # Return the loaded model

# Function to make predictions based on input data
def predict_sales(model, input_data):
    predictions = model.predict(input_data)  # Make predictions using the model
    return predictions[0]  # Return the prediction

# setting page configuration:
st.set_page_config(page_title="Retail Sales Prediction")

# setting the title:
st.title("Retail Sales Prediction")
st.write("---")

# setting Sidebar :
with st.sidebar:
    st.title("Retail Sales Dashboard")  

    select = option_menu(
        "Main Menu",
        options=["Home", "Price Prediction 1","Price Prediction 2" ,"Analysis", "Recommendation", "Conclusion"],
        icons=["house", "calculator", "calculator","graph-up", "stars", "check2-circle"], 
        menu_icon="cast",  # Menu icon at the top
        default_index=0,  # selected by default
        styles={
            "container": {"background-color": "#1f1f2e"},  # Darker background color
            "icon": {"color": "#FFD700", "font-size": "22px"},  # Gold color for icons
            "nav-link": {"font-size": "18px", "color": "#FFFFFF", "text-align": "left", "margin": "0px", "--hover-color": "#5a5a72"},  # White text color and hover effect
            "nav-link-selected": {"background-color": "#4CAF50", "color": "#FFFFFF"},  # Green background for selected item
        }
    )

if select == "Home":

    st.markdown("""
    ### **Introduction**
    Welcome to the **Retail Sales Prediction Project**! Our goal is to accurately forecast weekly sales for different departments across stores using advanced machine learning techniques. This project aims to empower retail businesses by providing actionable insights and optimizing strategies for inventory management, promotional planning, and resource allocation.
    """)

    st.markdown("""
    ### **Business Use Cases**
    - **Sales Forecasting**: Accurately predict future sales to manage inventory effectively and avoid overstock or stockout situations.
    - **Promotional Planning**: Evaluate the impact of markdowns and strategize future promotions to maximize sales.
    - **Holiday Strategy**: Develop data-driven strategies for peak holiday seasons to boost revenue.
    """)

    st.markdown("""
    ### **Data Set Overview**
    The project utilizes historical sales data from 45 stores, including:
    - **Stores Data**: Information on store types and sizes.
    - **Features Data**: External factors like temperature, fuel price, consumer price index (CPI), unemployment rate, and holiday indicators.
    - **Sales Data**: Weekly sales figures for various departments from 2010 to 2012.
    """)

    st.markdown("""
    ### **Approach**
    1. **Data Understanding and Preprocessing**: Cleaning and preparing data for analysis, including handling missing values and feature extraction.
    2. **Exploratory Data Analysis (EDA)**: Visualizing sales trends and uncovering patterns and seasonality.
    3. **Feature Engineering**: Creating new predictive features like "Day", "Month",etc.
    4. **Model Selection and Training**: Testing multiple models like Random Forest, and AdaBoost to find the best predictor.
    5. **Markdown Effect Analysis**: Quantifying the impact of markdowns on sales to inform future pricing strategies.
    """)

    st.write(
        """
        The **Retail Sales Prediction Project** aimed to forecast department-wise weekly sales for retail stores using historical sales data, enhancing decision-making processes across the organization. This project utilized a comprehensive approach involving multiple phases, from data understanding to model deployment, each contributing to the successful development of a robust sales prediction model.
        """
    )

if select == "Price Prediction 1":

    # Loading pre-trained model :
    model_path = r"C:\Users\ramya\Final_ML_Model_1.pkl"
    model = load_model(model_path)

    # Loading the data :
    df = pd.read_csv("C:\\Users\\ramya\\Final_Model1.csv")

    # Setting the title :
    st.title("Sales Prediction With MarkDown")

    col1, col2 = st.columns(2)  
    with col1:
       
        store = st.selectbox("Select Store", df["Store"].unique())
        dept = st.selectbox("Select Department", df["Dept"].unique())
        day = st.selectbox("Enter Day of the Week", df["Day"].unique())
        month = st.selectbox("Enter Month", df["Month"].unique())
        year = st.number_input("Enter Year", min_value=2000, max_value=2100)
        size = st.number_input("Enter Store Size / Min:34875, Max:219622")
        temperature = st.number_input("Enter Temperature / Min:-7.29, Max:101.95")
        fuel_price = st.number_input("Enter Fuel Price / Min:2.472, Max:4.468")
        cpi = st.number_input("Enter CPI / Min:126.064, Max:228.9764563")

    with col2:
        is_holiday = st.selectbox("Is Holiday", ["No", "Yes"])
        type_ = st.selectbox("Select Store Type", df["Type"].unique())
        unemployment = st.number_input("Enter Unemployment Rate / Min:3.684, Max:14.313")    
        markdown1 = st.number_input("Enter MarkDown1 / Min:-2781.45 Max:103184.98")
        markdown2 = st.number_input("Enter MarkDown2 / Min:-265.76, Max:104519.54")
        markdown3 = st.number_input("Enter MarkDown3 / Min:-179.26, Max:149483.31")
        markdown4 = st.number_input("Enter MarkDown4 /Min:0.22, Max:67474.85")
        markdown5 = st.number_input("Enter MarkDown5 /Min:-185.17, Max:771448.1")
        customer = st.number_input("Enter Estimated Customer Count")

    # Preparing the input data for prediction by creating a DataFrame :
    input_data = pd.DataFrame({ "Day": [day],
                                "Month": [month],
                                "Year": [year],
                                "Store": [store],
                                "Dept": [dept],
                                "Type": [type_],
                                "Size": [size],
                                "IsHoliday": [1 if is_holiday == "Yes" else 0],
                                "Temperature": [temperature],
                                "Fuel_Price": [fuel_price],
                                "MarkDown1": [markdown1],
                                "MarkDown2": [markdown2],
                                "MarkDown3": [markdown3],
                                "MarkDown4": [markdown4],
                                "MarkDown5": [markdown5],
                                "CPI": [cpi],
                                "Unemployment": [unemployment],
                                "Customer":[customer] })


    # Button for prediction :
    if st.button("Predict Sales", use_container_width= True):
        price = predict_sales(model, input_data)  # Calling the prediction function
        st.balloons()
        st.success(f"Predicted Weekly Sales: ${price:,.2f}")  # Display the predicted weekly sales


if select == "Price Prediction 2":
    
    st.title("Sales Prediction without Markdown ")    

    # Loading pre-trained model :
    model_path = r"C:\Users\ramya\Final_ML_Model_2.pkl"
    model = load_model(model_path)

    # Loading the data :
    df = pd.read_csv(r"C:\Users\ramya\Final_Model2.csv")

    col1, col2= st.columns(2)  
    with col1:
       
        store = st.selectbox("Select Store", df["Store"].unique())
        dept = st.selectbox("Select Department", df["Dept"].unique())
        day = st.selectbox("Enter Day of the Week", df["Day"].unique())
        month = st.selectbox("Enter Month", df["Month"].unique())
        year = st.number_input("Enter Year", min_value=2000, max_value=2100)
        size = st.number_input("Enter Store Size / Min:34875, Max:219622")
        temperature = st.number_input("Enter Temperature / Min:-7.29, Max:101.95")

    with col2:
        fuel_price = st.number_input("Enter Fuel Price / Min:2.472, Max:4.468")
        cpi = st.number_input("Enter CPI / Min:126.064, Max:228.9764563")
        is_holiday = st.selectbox("Is Holiday", ["No", "Yes"])
        type = st.selectbox("Select Store Type", df["Type"].unique())
        unemployment = st.number_input("Enter Unemployment Rate / Min:3.684, Max:14.313")    
        customer = st.number_input("Enter Estimated Customer Count")

    # Prepare input data for prediction by creating a DataFrame :
    input_data = pd.DataFrame({ "Day": [day],
                                "Month": [month],
                                "Year": [year],
                                "Store": [store],
                                "Dept": [dept],
                                "Type": [type],
                                "Size": [size],
                                "IsHoliday": [1 if is_holiday == "Yes" else 0],
                                "Temperature": [temperature],
                                "Fuel_Price": [fuel_price],
                                "CPI": [cpi],
                                "Unemployment": [unemployment],
                                "Customer":[customer]})

    # Button for prediction
    if st.button("Predict Sales", use_container_width = True):
        price = predict_sales(model, input_data)  # Calling the prediction function
        st.balloons()
        st.success(f"Predicted Weekly Sales: ${price:,.2f}")  # Displaying the predicted weekly sales

if select == "Analysis" :

    # Loading the data
    df_m3 = pd.read_csv("C:\\Users\\ramya\\Final_M1.csv")

    st.title("Analysis of Markdown Impact on Sales")

    markdown_types = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

    # Correlation Analysis
    st.subheader("Correlation Analysis Between Markdowns and Weekly Sales")

    for markdown_type in markdown_types:
        correlation = df_m3[markdown_type].corr(df_m3["Weekly_Sales"])
        st.write(f"**Correlation between {markdown_type} and Weekly Sales: {correlation:.4f}**")

    st.markdown("---")
    st.header("Overall Analysis of All Markdown Results")

    st.subheader("Weak Correlation:")
    st.write("All markdown types show weak positive correlations with weekly sales, suggesting that markdowns have a limited effect on increasing sales.")

    st.subheader("Minimal Impact:")
    st.write("The small magnitude of these correlations implies that markdowns alone are not a significant driver of sales changes. Other factors are likely more influential in determining weekly sales performance.")

    st.markdown("---")
    # Analysis of Each Markdown Type:
    for markdown_type in markdown_types:
        st.subheader(f"{markdown_type} Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Plotting the distribution of Markdown values:
            st.write(f"**Distribution of {markdown_type}:**")
            fig = plt.figure(figsize=(6, 5))
            sns.histplot(df_m3[markdown_type], kde=True, bins=30, color='green')
            plt.title(f'Distribution of {markdown_type}')
            plt.grid(True)
            st.pyplot(fig)
        with col2:
            st.write(f"**{markdown_type} vs Weekly Sales:**")
            fig = plt.figure(figsize=(6, 5))
            sns.scatterplot(x = df_m3[markdown_type], y = df_m3["Weekly_Sales"], color='red')
            plt.title(f'{markdown_type} vs Weekly Sales')
            plt.grid(True)
            st.pyplot(fig)
            
    st.write("---")
    st.header("Overall Analysis of All Markdown Results")
    st.subheader("Left-Skewed Distributions:")
    st.write("""
            1. Across all markdowns (MarkDown1 to MarkDown5), the histograms show a consistent pattern of being heavily skewed to the left. The majority of markdown values are concentrated at the lower end of the scale, close to zero. This indicates that smaller markdowns are more frequently applied across different markdown categories.

            2. The frequency of predictions sharply declines as markdown values increase, suggesting that larger markdowns are rare.
                """)
    st.subheader("Weak Correlation with Weekly Sales:")
    st.write("""
            1. The scatter plots for each markdown versus weekly sales show a high variability in sales at lower markdown values. This implies that while markdowns are commonly applied at lower levels, they do not consistently lead to a proportional increase in weekly sales.
                
            2. Higher markdowns do not show a clear, consistent relationship with higher sales. In many cases, significant markdowns are associated with a wide range of sales figures, indicating that markdowns are not the sole or dominant factor influencing weekly sales.
                """)
                
if select == "Recommendation":

    st.markdown("---")

    model_markdown1 = load_model(r"C:\Users\ramya\Markdown_1.pkl")
    model_markdown2 = load_model(r"C:\Users\ramya\Markdown_2.pkl")
    model_markdown3 = load_model(r"C:\Users\ramya\Markdown_3.pkl")
    model_markdown4 = load_model(r"C:\Users\ramya\Markdown_4.pkl")
    model_markdown5 = load_model(r"C:\Users\ramya\Markdown_5.pkl")

    # Loading the data
    df_m3 = pd.read_csv("C:\\Users\\ramya\\Final_M1.csv")

    # Displaying Markdown Insights
    markdown_types = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

    st.title("Analysis of Markdown Impact on Sales")

    df_H = df_m3[df_m3["IsHoliday"] == 1]
    df_NH = df_m3[df_m3["IsHoliday"] == 0]

    # Mean Markdown values
    markdown_means_holiday = df_H[markdown_types].mean()
    markdown_means_nonholiday = df_NH[markdown_types].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    markdown_means = pd.DataFrame({"Holiday": markdown_means_holiday,
                                   "Non-Holiday": markdown_means_nonholiday}).T
    markdown_means.plot(kind='bar', ax = ax)
    ax.set_title('Average Markdown Values by Holiday Status')
    ax.grid(True)
    st.pyplot(fig)

    st.title("Recommendations")
    
    with st.expander("**Increase Markdown Levels During Holidays**", expanded=True):
        st.write("""
        - **Insight:** Data indicates that markdown levels are generally higher during holidays compared to non-holidays. This suggests a strong correlation between markdowns and increased sales during these peak periods.
        - **Action:** To maximize sales during holidays, consider implementing higher markdowns on popular products. 
        """)

    with st.expander("**Optimize Markdown Allocations**"):
        st.write("""
        - **Insight:** The effectiveness of markdowns can vary across different departments and stores. Some locations or product categories may benefit more from markdowns than others.
        - **Action:** Analyze historical sales data to identify which departments or stores experience the highest increase in sales with markdowns. Focus markdown efforts on these areas to maximize return on investment.
        """)

    with st.expander("**Seasonal and Promotional Analysis**"):
        st.write("""
        - **Insight:** The impact of markdowns can also be influenced by seasonal trends and specific promotional events.
        - **Action:** Conduct a seasonal analysis to understand how markdowns perform throughout the year and during different promotional events. 
        """)

    with st.expander("**Monitor and Adjust**"):
        st.write("""
        - **Insight:** The effectiveness of markdown strategies can change over time due to shifting market conditions and consumer behavior.
        - **Action:** Continuously monitor sales data and the impact of markdowns to identify trends and make adjustments as needed. 
        """)

    with st.expander("**Customer Feedback Integration**"):
        st.write("""
        - **Insight:** Customer responses to markdowns can provide valuable insights into their preferences and buying behavior.
        - **Action:** Collect and analyze customer feedback regarding markdowns and promotions. Use this feedback to refine markdown strategies and improve the overall customer experience.
        """)

if select == "Conclusion":

    st.markdown("---")
    st.header("Conclusion:")
    st.write(
        """
        The **Retail Sales Prediction Project** aimed to forecast department-wise weekly sales for retail stores using historical sales data, enhancing decision-making processes across the organization. This project utilized a comprehensive approach involving multiple phases, from data understanding to model deployment, each contributing to the successful development of a robust sales prediction model.
        """)
    st.header("Steps Involved")

    st.subheader("Data Understanding and Preprocessing")
    st.write(
        """
        - I started by thoroughly examining the provided datasets, which included information on store characteristics, sales figures, promotional markdowns, and economic indicators like CPI and unemployment rates.
        - Essential preprocessing steps, such as handling missing values, converting date formats, and normalizing numerical data, were performed to ensure data quality and consistency.
        """
    )

    st.subheader("Exploratory Data Analysis (EDA)")
    st.write(
        """
        - Through EDA, we visualized trends and patterns, identifying markdown predictions and holiday effects in the sales data.
        - Correlation analysis helped us understand the relationships between various features, such as the impact of fuel prices and regional economic conditions on sales.
        """
    )

    st.subheader("Feature Engineering")
    st.write(
        """
        - Categorical variables were encoded appropriately to ensure compatibility with machine learning algorithms.
        - Feature extraction was also conducted to derive new variables like year, month, week number, and Customer, which provided additional insights.

        """
    )

    st.subheader("Model Selection and Training")
    st.write(
        """
        - We experimented with various models , tree-based models (Random Forest) ,AdaBoost, GradientBoosting and XGB to identify the most accurate predictor of sales.
        - Data was split into training and validation sets to evaluate model performance using metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) and R2 Score.
        """
    )

    st.subheader("Markdown Effect Analysis")
    st.write(
        """
        - By simulating sales with  markdowns, we quantified the impact of promotional markdowns on sales.
        - This analysis provided actionable insights into optimizing pricing strategies and promotional planning to maximize sales.
        """
    )

    st.subheader("Model Evaluation and Validation")
    st.write(
        """
        - We tested the models using various metrics to ensure their robustness and generalizability.
        - The final model selection was based on its ability to accurately predict sales, particularly during holiday weeks, which are critical for business strategy.
        """
    )

    st.subheader("Deployment and Recommendations")
    st.write(
        """
        - The best-performing model was deployed and allowing for real-time predictions and scalability.
        - We provided strategic recommendations for optimizing markdown strategies, management, and resource allocation based on the model's insights.
        """
    )


    

