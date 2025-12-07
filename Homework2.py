# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "kagglehub",
#     "marimo",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Importing The Required Libraries
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import kagglehub
    import os 
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    return (
        KMeans,
        StandardScaler,
        accuracy_score,
        classification_report,
        confusion_matrix,
        kagglehub,
        np,
        os,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Our Functions
    """)
    return


@app.cell
def _(pd):
    def CorrectDate(df):
        for col in df.columns:
            if 'date' in col:
                df[col] = pd.to_datetime(df[col])
    return (CorrectDate,)


@app.cell
def _(StandardScaler, train_test_split):
    def Split_and_scale(X, y, test_size=0.2, scaled=True):
        _X_train, _X_test, _y_train, _y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Splitting the data into training and testing sets
        if scaled == True:
            scaler = StandardScaler()
            _X_train_scaled = scaler.fit_transform(_X_train)  # Returning scaled or unscaled data based on the Scaled parameter
            _X_test_scaled = scaler.transform(_X_test)
            return (_X_train_scaled, _X_test_scaled, _y_train, _y_test)
        else:
            return (_X_train, _X_test, _y_train, _y_test)
    return


@app.cell
def _(accuracy_score, classification_report, confusion_matrix):
    def Evaluate_model(y_test, y_pred):
        print('Accuracy:', accuracy_score(_y_test, _y_pred))
        print(confusion_matrix(_y_test, _y_pred))
        print(classification_report(_y_test, _y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Getting Our Data
    """)
    return


@app.cell
def _(kagglehub):
    # Download the dataset from Kaggle
    path = kagglehub.dataset_download("ivanchvez/99littleorange")
    print("Path to dataset files:", path)
    return (path,)


@app.cell
def _(os, path):
    # Check current working directory and change to dataset path
    print(os.getcwd())
    os.chdir(path)

    # List files that has been downloaded
    os.listdir()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visual Looking To Our Datasets
    """)
    return


@app.cell
def _(pd):
    df_calendar = pd.read_csv('calendar.csv')
    df_calendar
    return (df_calendar,)


@app.cell
def _(pd):
    df_city = pd.read_csv('city.csv')
    df_city
    return (df_city,)


@app.cell
def _(pd):
    df_passenger= pd.read_csv('passenger.csv')
    df_passenger
    return (df_passenger,)


@app.cell
def _(pd):
    df_trip = pd.read_csv('trip.csv')
    df_trip
    return (df_trip,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Preparation: Cleaning & Preprocessing
    """)
    return


@app.cell
def _(df_calendar, df_city, df_passenger):
    # Rename columns to add prefixes
    df_city.columns  = 'city_' + df_city.columns 
    df_passenger.columns  = 'passenger_' + df_passenger.columns
    df_calendar.columns  = 'calendar_' + df_calendar.columns

    # This will rename a specific column back to its original name
    df_calendar.rename(columns={'calendar_calendar_date': 'calendar_date'}, inplace=True)
    return


@app.cell
def _(df_city, df_passenger, df_trip):
    # Adding the city names and first_call_time to our trips dataframe
    # by merging on df_trip, df_city, and df_passenger
    df = df_trip.merge(df_city, on='city_id', how='left')
    df = df.merge(df_passenger, on='passenger_id', how='left')

    # Let's see the shape of the dataframe after merging
    print(df.info(), '\n-------------\n')
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > As we see, the time columns are strings instead of datetime
    """)
    return


@app.cell
def _(df, df_calendar, pd):
    # Let's correct time and date columns
    df['call_time'] = pd.to_datetime(df['call_time'], format="%m/%d/%Y %I:%M:%S %p")
    df['finish_time'] = pd.to_datetime(df['finish_time'], format="%m/%d/%Y %I:%M:%S %p")
    df['passenger_first_call_time'] = pd.to_datetime(df['passenger_first_call_time'], format="%m/%d/%Y %I:%M:%S %p")
    df['Trip_duration'] = (df['finish_time'] - df['call_time']).dt.total_seconds() / 60
    df_calendar['calendar_date'] = pd.to_datetime(df_calendar['calendar_date'], format="%m/%d/%Y")
    return


@app.cell
def _(CorrectDate, df):
    # Extract date from datetime columns
    df['passenger_first_call_date'] = df['passenger_first_call_time'].dt.date
    df['call_date'] = df['call_time'].dt.date

    # The new columns will be of object type, so we convert them to datetime
    CorrectDate(df)

    # Let's see the final structure of our dataframe
    print(df.columns)
    print('\n-------------------\n')
    print(df.info())
    return


@app.cell
def _(CorrectDate, df, pd):
    # The current dataframe 'df' contains trip-level data.
    # We will now aggregate this data to create a user-level dataframe 'df_users'.
    df_users = pd.pivot_table(
        df,
        index=['passenger_id'],
        aggfunc={
            'id': 'nunique',
            'city_id': 'first',
            'city_name': 'nunique',
            'driver_id': 'nunique',                  
            'surge_rate': 'mean',                    
            'trip_distance': ['sum', 'mean'],        
            'trip_fare': ['sum', 'mean'],            
            'passenger_first_call_date': 'first',    
            'call_date': 'last'                
        }
    )

    print('Old Columns', df_users.columns)

    # Flattening and renaming multiIndex columns
    df_users.columns = [
        'last_trip_date',
        'first_city',             
        'distinct_cities',          
        'distinct_drivers',     
        'total_trips',                
        'first_trip_date',            
        'avg_surge',               
        'avg_trip_distance',          
        'total_trip_distance',        
        'avg_trip_fare',            
        'total_trip_fare'      
    ]

    # Changing date columns to datetime format
    CorrectDate(df_users)

    # Let's reset the index to have passenger_id as a column
    df_users.reset_index(inplace=True)

    # Final structure of user-level dataframe
    df_users.info()
    return (df_users,)


@app.cell
def _(churn_threshold_slider, df, df_users):
    # According to the problem definition, a customer is considered churned if they haven’t 
    # used the application for more than 28 days.
    # Let's get the churn threshold date
    all_dates = df[['call_date']].sort_values(by='call_date')['call_date'].unique()
    max_date = all_dates[-1]
    steps_back = churn_threshold_slider.value + 1
    churn_cutoff = all_dates[-steps_back]
    mau_index = df_users[df_users['last_trip_date'] < churn_cutoff].index
    df_users['Churn'] = 0
    df_users.loc[mau_index, 'Churn'] = 1
    churn_rate = df_users['Churn'].mean()
    return (churn_cutoff,)


@app.cell
def _(df_calendar, df_users):
    # The only columns I want from calendar are calendar_date and calendar_holiday
    df_calendar_1 = df_calendar[['calendar_date', 'calendar_holiday']]
    df_users_1 = df_users.merge(df_calendar_1, left_on=df_users['first_trip_date'], right_on=df_calendar_1['calendar_date'], how='left')
    # Merge to df_users and df_calendar to get which customers joined to us on a holiday
    df_users_1['Joined On Holiday'] = df_users_1['calendar_holiday']
    df_users_1['Churn'] = df_users['Churn'] 
    df_users_1.drop(columns=['calendar_date', 'calendar_holiday', 'key_0'], inplace=True)
    # Droping unnecessary columns 
    df_users_1.dropna(inplace=True)
    # We have some NaN values in avg_surge and trip_distance columns
    # They are a few, so let's drop them
    # Final structure of user-level dataframe
    df_users_1.info()
    return (df_users_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ML Model Part
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature Engineering Part
    """)
    return


@app.cell
def _(churn_cutoff, df_users_1, np):
    # Creating our new features
    df_users_1['Customer_Lifetime_Days'] = (churn_cutoff - df_users_1['first_trip_date']).dt.days
    df_users_1["trips_per_week"] = df_users_1["total_trips"] / ((df_users_1["Customer_Lifetime_Days"] / 7) + 1)
    df_users_1['days_since_last_trip'] = df_users_1['Customer_Lifetime_Days'] - df_users_1['last_trip_date'].dt.dayofyear
    df_users_1['trips_per_day'] = df_users_1['total_trips'] / (df_users_1['Customer_Lifetime_Days'] + 1)
    df_users_1['fare_per_km'] = df_users_1['total_trip_fare'] / (df_users_1['total_trip_distance'] + 1)
    df_users_1['fare_per_trip'] = df_users_1['total_trip_fare'] / (df_users_1['total_trips'] + 1)
    df_users_1['surge_volatility'] = df_users_1['avg_surge'] * df_users_1['distinct_cities']
    df_users_1['long_trip_ratio'] = df_users_1['total_trip_distance'] / (df_users_1['avg_trip_distance'] + 1)
    df_users_1['lifetime_trip_rate'] = df_users_1['total_trips'] / (df_users_1['Customer_Lifetime_Days'] + 1)
    df_users_2 = df_users_1.replace([np.inf, -np.inf], np.nan)
    # Corecting inf due to dividion by zero
    # We have some NaN values after creating new features
    df_users_2.dropna(inplace=True)
    df_users_1
    return (df_users_2,)


@app.cell
def _(df_users_2, plt, sns):
    new_features = ['days_since_last_trip', 'trips_per_day', 'trips_per_week', 'fare_per_km', 'fare_per_trip', 'surge_volatility', 'long_trip_ratio', 'lifetime_trip_rate', 'Churn']
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_users_2[new_features].corr(), annot=False)
    plt.title('Correlation Heatmap')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Clustering Part
    """)
    return


@app.cell
def _(StandardScaler, df_users_2):
    # Our Chosed features
    Cluster_f = ['Customer_Lifetime_Days', 'trips_per_week', 'lifetime_trip_rate', 'fare_per_trip', 'total_trip_fare', 'long_trip_ratio', 'distinct_cities']
    cluster_features = df_users_2[Cluster_f].copy()
    scaler = StandardScaler()
    # Getting a copy of our data with the most important features
    # Scale features
    X_scaled = scaler.fit_transform(cluster_features)
    return X_scaled, cluster_features, scaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Let's get the best number of clusters for our data
    """)
    return


@app.cell
def _(KMeans, X_scaled, plt):
    inertias = []
    K = range(2, 8)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(K, inertias, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(KMeans, X_scaled, df_users_2, n_clusters_slider):
    best_k = n_clusters_slider.value
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    _cluster_labels = kmeans_final.fit_predict(X_scaled)
    df_users_2['cluster_behavior'] = _cluster_labels
    return (kmeans_final,)


@app.cell
def _(cluster_features, kmeans_final, pd, scaler):
    # Let's get the centers and figure out the charaterics of our clusters
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans_final.cluster_centers_),
        columns=cluster_features.columns
    )

    centers
    return (centers,)


@app.cell
def _(df_users_2):
    # Renaming our clusters
    df_users_2['cluster_behavior'].replace({0: 'Casual Customer', 1: 'High-usage customers', 2: 'Moderate-usage customers'}, inplace=True)
    # Showing different clusters of our current users
    df_users_2[df_users_2['Churn'] == 0]['cluster_behavior'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > The analysis shows that the segments worth focusing on are the moderate-usage customers with an average lifetime near to 306 days, and the high-usage customers near to 282 days. The other clusters represent casual users whose churn is less critical.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # The Visualization Part
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    churn_threshold_slider = mo.ui.slider(
        start=28, stop=180, value=28,
        label="Churn inactivity threshold (days)"
    )

    n_clusters_slider = mo.ui.slider(start=3, stop=12, step=1, label="Number of clusters")


    trips_week_slider = mo.ui.slider(
        start=1, stop=10, value=1,
        label="Trips per week"
    )
    return churn_threshold_slider, n_clusters_slider, trips_week_slider


@app.cell
def _(df_users_2, mo):
    date_from = mo.ui.date(
        label="From",
        value=str(df_users_2['last_trip_date'].min().date())
    )

    apply_button = mo.ui.button("Apply Date Range")

    date_to = mo.ui.date(
        label="To",
        value=str(df_users_2['last_trip_date'].max().date())
    )

    apply_button = mo.ui.button("Apply Date Range")
    return apply_button, date_from, date_to


@app.cell
def _(
    apply_button,
    churn_threshold_slider,
    date_from,
    date_to,
    mo,
    n_clusters_slider,
    trips_week_slider,
):
    mo.vstack([
        churn_threshold_slider,
        n_clusters_slider,
        trips_week_slider,
        mo.hstack([date_from, date_to, apply_button])
    ])
    return


@app.cell
def _(date_from, date_to, df_users_2):
    df_users_3 = df_users_2[df_users_2['last_trip_date'].between(str(date_from.value),str(date_to.value))]
    return (df_users_3,)


@app.cell
def _(churn_cutoff, churn_threshold_slider, df_users_3, mo):
    churned = df_users_3["Churn"].sum()
    churn_rate_ = df_users_3["Churn"].mean()


    last_date_ = df_users_3.sort_values('last_trip_date', ascending = False)['last_trip_date'].unique()[:churn_threshold_slider.value][-1]
    active_users = df_users_3[df_users_3['last_trip_date']> last_date_]["passenger_id"].nunique()


    card1 = mo.md(
            f"""
            ### Churn cutoff date  
            **{churn_cutoff}**
            """
        )

    card2 = mo.md(
            f"""
            ### Churned users  
            **{churned:,}**
            """
        )

    card3 = mo.md(
            f"""
            ### Churn rate  
            **{churn_rate_:.2%}**
            """
        )

    card4 = mo.md(
            f"""
            ### Active Users  
            **{active_users:,}**
            """
        )

    # Display them beside each other (like Power BI)
    mo.hstack([card1, card2, card3, card4], widths="equal", gap=2)
    return


@app.cell
def _(centers, np, plt):
    # Getting feature columns from the cluster centers.
    _features = list(centers.columns)
    data = centers.values
    # Getting the centers values
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    # Normalize Data
    # We scale all feature values to the range 0.1–1.0 so the radar
    norm_data = (data - data_min) / (data_max - data_min + 1e-09)
    norm_data = 0.1 + 0.9 * norm_data
    plot_data = np.concatenate([norm_data, norm_data[:, [0]]], axis=1)
    num_vars = data.shape[1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    # Prepare Data for Radar Chart
    angles = np.concatenate([angles, [angles[0]]])
    _cluster_labels = [f"Cluster {i}" for i in range(centers.shape[0])]
    # Create Angles for Each Feature
    #  Angles are evenly spaced around the circle.
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for i, label in enumerate(_cluster_labels):  # close the loop
        ax.plot(angles, plot_data[i], linewidth=2, marker='o', label=label)
    # Plot Radar Chart
        ax.fill(angles, plot_data[i], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(_features)
    ax.set_ylim(0, 1)
    plt.title('Customer Loyalty Cluster Profile (Radar Chart)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    # Format Axes and Labels
    plt.show()
    return


@app.cell
def _(df_users_3, plt):
    def plot_feature_bar_matplotlib(df = df_users_3, feature = 'cluster_behavior' ):
        df_grouped = df.groupby("cluster_behavior")[feature].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df_grouped.index, df_grouped[feature])
        ax.set_title(f"Bar Chart for {feature}")
        ax.set_xlabel("Index")
        ax.set_ylabel(feature)
        plt.tight_layout()

        return fig
    return


@app.cell
def _(df_users_2, mo):
    cols = df_users_2.columns.tolist()

    feature_select1 = mo.ui.dropdown(
        options=cols,
        label="Choose a feature for the barplot",
        value=cols[0]
    )


    feature_select2 = mo.ui.dropdown(
        options=cols,
        label="Choose x feature to the barplot",
        value=cols[0]
    )
    mo.hstack([feature_select1, feature_select2])
    return


@app.cell
def _(df_users_3):
    def countplot_seaborn(df, feature):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 4))

        # Orange countplot
        sns.countplot(
            data=df,
            x=feature,
            color="#FF8C00"
        )

        plt.title(f"Countplot of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Number Of Users")
        plt.xticks(rotation=45)
        plt.tight_layout()

        fig = plt.gcf()
        return fig



    countplot_seaborn(df_users_3, "cluster_behavior")
    return


@app.cell
def _(df_users_3):
    df_users_3.select_dtypes("object").columns
    return


@app.cell
def _(df_users_3):
    df_users_3
    return


@app.function
def build_pivot_table(df, index_col, value_col, aggfunc_name):
    import pandas as pd

    # Map readable names to real pandas aggfuncs
    agg_map = {
        "Mean": "mean",
        "Sum": "sum",
        "Count": "count",
        "Std": "std",
        "Min": "min",
        "Max": "max",
    }

    aggfunc = agg_map[aggfunc_name]

    pivot = pd.pivot_table(
        df,
        index=index_col,
        values=value_col,
        aggfunc=aggfunc,
    )

    return pivot.reset_index()


@app.cell
def _():
    def _(df_users_2, mo):
        import pandas as pd

        # candidates for index (rows): typically categorical
        index_cols = df_users_2.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if not index_cols:
            index_cols = df_users_2.columns.tolist()  # fallback

        # candidates for values: numeric features
        value_cols = df_users_2.select_dtypes(include="number").columns.tolist()

        aggfunc_options = ["Mean", "Sum", "Count", "Std", "Min", "Max"]

        index_select = mo.ui.dropdown(
            options=index_cols,
            label="Index (rows)",
            value=index_cols[0],
        )

        value_select = mo.ui.dropdown(
            options=value_cols,
            label="Feature (values)",
            value=value_cols[0],
        )

        aggfunc_select = mo.ui.dropdown(
            options=aggfunc_options,
            label="Aggregation",
            value="Mean",
        )

        # Show the controls
        mo.hstack([index_select, value_select, aggfunc_select])

        return index_select, value_select, aggfunc_select
    return


@app.cell
def _():
    def _(df_users_2, index_select, value_select, aggfunc_select, build_pivot_table, mo):
        index_col = index_select.value
        value_col = value_select.value
        aggfunc_name = aggfunc_select.value

        pivot = build_pivot_table(
            df_users_2,
            index_col=index_col,
            value_col=value_col,
            aggfunc_name=aggfunc_name,
        )

        mo.vstack(
            [
                mo.md(
                    f"### Pivot table\n"
                    f"- **Index:** `{index_col}`  \n"
                    f"- **Feature:** `{value_col}`  \n"
                    f"- **Aggfunc:** `{aggfunc_name}`"
                ),
                mo.ui.table(pivot),
            ]
        )

        return pivot
    return


@app.cell
def _(df_users_3, mo):
    multi_features = mo.ui.multiselect(
        options=df_users_3.columns.tolist(),
        label="Choose multiple columns",
        value=[],      
        )

    multi_index = mo.ui.multiselect(
        options=df_users_3.select_dtypes("object").columns,
        label="Choose multiple columns",
        value=[],      
        )

    mo.vstack([multi_index, multi_features, ])
    return (multi_index,)


@app.cell
def _(df_users_3, multi_index, pd):
    pd.pivot_table(
        df_users_3,
        index=multi_index.value,
        columns=multi_index.value,
        values=None,
        aggfunc="size"
    )
    return


@app.cell
def _(df_city, df_users_3):
    df_users_3['first_city'] = df_users_3['first_city'].map(
        df_city.set_index('city_id')['city_name']
    )
    return


@app.cell
def _(multi_index):
    multi_index.value
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
