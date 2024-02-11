import streamlit as st
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
# streamlit cache


st.markdown('# Activity 1 Dashboard: Cooling Load Prediction With Linear Regression')
st.markdown("""Cooling load refers to the amount of heat that must be removed from a space or a building to maintain 
a comfortable (20-25C) and controlled indoor environment.""")

with st.spinner('Creating Correlation Matrix...'):
    from ucimlrepo import fetch_ucirepo
    import pandas as pd
    energy_efficiency = fetch_ucirepo(id=242)

    X = pd.read_csv('x_values.csv')
    y = pd.read_csv('y_values.csv')

    new_columns = {}
    for row in energy_efficiency.variables.itertuples():
        new_columns[getattr(row, 'name')] = getattr(row, 'description')

    X = X.rename(columns=new_columns)
    y = y.rename(columns=new_columns)
    df = X
    df['Heating Load'] = y['Heating Load']
    df['Cooling Load'] = y['Cooling Load']
    st.markdown('### Correlation Matrix of the Features and Target Variable')
    st.table(df.corr())

dark_mode_css = """
    <style>
        body {
            color: white;
            background-color: #2E2E2E;
        }
    </style>
"""
st.markdown(dark_mode_css, unsafe_allow_html=True)

# Setting seaborn style to 'dark' to complement Streamlit's dark mode
sns.set_style('dark')

# Neon color palette for the plots
neon_palette = ["#FFF3CF", "#FF6EC7"]
sns.set_palette(neon_palette)

# EDA with Regression Plots title
st.markdown("### EDA (Exploratory Data Analysis) With Regression Plots")

# Assuming 'df' is your DataFrame and is already loaded
# Replace 'df', 'Cooling Load', and 'Roof Area' with your actual DataFrame and columns

with st.spinner('Creating Regression Plots...'):
    # Creating a figure and adjusting its background to match Streamlit's dark mode
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#2E2E2E')  # Set the outer color
    ax.set_facecolor('#2E2E2E')  # Set the graph background color

    # Adjusting plot text and edge colors for visibility
    ax.tick_params(colors='white', which='both')  # Change the color of ticks
    for spine in ax.spines.values():
        spine.set_edgecolor('white')  # Change the color of the plot's spines

    # Creating the regression plot
    sns.regplot(data=df, y='Cooling Load', x='Roof Area', ax=ax)

    # Adjust labels if necessary
    ax.set_xlabel('Roof Area', color='white')
    ax.set_ylabel('Cooling Load', color='white')

    # Finally, showing the plot in Streamlit
    st.pyplot(fig)
    plt.figure()



with st.spinner("Training the linear regression model with the label=`Cooling Load` and feature=`Roof Area`"):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers.legacy import Adam
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test =train_test_split(df['Roof Area'], df['Cooling Load'], test_size=0.2)

    reg_model = Sequential()
    reg_model.add(
        Dense(units=1, input_dim=1, activation='linear')
    )
    reg_model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.5))
    history = reg_model.fit(
        X_train, y_train,  batch_size=10000, epochs=5500)

    st.markdown('### Training and Validation Loss')
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#2E2E2E')  # Set the outer color
    ax.set_facecolor('#2E2E2E')  # Set the graph background color

    # Adjusting plot text and edge colors for visibility
    ax.tick_params(colors='white', which='both')  # Change the color of ticks
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    plt.plot(history.history['loss'], label='Training Loss', color='orange')
    plt.title('Training and Validation Loss', color='white')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)
    plt.figure()


    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#2E2E2E')  # Set the outer color
    ax.set_facecolor('#2E2E2E')  # Set the graph background color

    # Adjusting plot text and edge colors for visibility
    ax.tick_params(colors='white', which='both')  # Change the color of ticks
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    st.markdown('### Model Evaluation: Predicted VS. Actual')
    y_hat = reg_model.predict(X_test)
    plt.scatter(X_test.squeeze(), y_test, label='Actual (y_test)', color='pink')
    plt.plot(X_test.squeeze(), y_hat, label='Predicted (y_hat)', color='orange')
    plt.xlabel('X_test')
    plt.ylabel('Values')
    plt.title('Comparison of Actual and Predicted Values')
    plt.legend()
    st.pyplot(plt)
    plt.figure()

st.markdown("## Additional Info: Multi-Variable Linear Regression")
st.markdown("Let's try using multiple values to predict the cooling load. We will use the following features: `Surface Area`, `Wall Area`, `Roof Area`, `Overall Height`")
# Let's use scikit-learn to train a multi-variable linear regression model
import numpy as np
with st.spinner('Training a multi-variable linear regression model...'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression


    sns.set(style="darkgrid")  # Set seaborn style to dark to better fit the dark theme
    plt.rcParams['figure.facecolor'] = '#2E2E2E'  # Outer color
    plt.rcParams['axes.facecolor'] = '#2E2E2E'  # Axes background color
    plt.rcParams['savefig.facecolor'] = '#2E2E2E'  # Background color of the saved files
    plt.rcParams['text.color'] = 'white'  # Setting text color to white for visibility
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    y_trimmed = y['Cooling Load']
    X_trimmed = X[['Roof Area', 'Surface Area', 'Overall Height', 'Wall Area']]
    X_train, X_test, y_train, y_test = train_test_split(X_trimmed, y_trimmed, test_size=0.2)

    X_train, X_test, y_train, y_test = train_test_split(X_trimmed, y_trimmed, test_size=0.2, random_state=42)

    multi_reg = LinearRegression()
    multi_reg.fit(X_train, y_train)
    coef = multi_reg.coef_
    # create a df out of the coef and intercept
    stat = {}
    for i, data in enumerate(['Roof Area', 'Surface Area', 'Overall Height', 'Wall Area']):
        stat[data] = [coef[i]]
    coef_df = pd.DataFrame(stat, columns=['Roof Area', 'Surface Area', 'Overall Height', 'Wall Area'], index=['Coefficient'])

    st.markdown('### Coefficients of the Multi-Variable Linear Regression Model')
    st.table(coef_df)

    y_hat = multi_reg.predict(X_test)
    fig.patch.set_facecolor('#2E2E2E')  # Set the outer color
    ax.set_facecolor('#2E2E2E')


    X_test_with_predictions = X_test.copy()
    X_test_with_predictions['Cooling Load Actual'] = y_test
    X_test_with_predictions['Cooling Load Predicted'] = y_hat

    sns.pairplot(X_test_with_predictions,
                 diag_kind='kde')

    st.pyplot(plt)



