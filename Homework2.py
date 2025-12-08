import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os 
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    import marimo as mo
    return (
        KMeans,
        StandardScaler,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mo,
        np,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Importing The Required Libraries
    """) if mo.app_meta().mode == "edit" else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Our Functions
    """) if mo.app_meta().mode == "edit" else None
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


@app.cell
def _(plt, sns):
    def countplot_seaborn(df, feature):
        plt.figure(figsize=(6, 4))

        # Compute counts and sort them
        order = df[feature].value_counts().sort_values(ascending=False).index

        sns.countplot(
            data=df,
            x=feature,
            order=order,          # <-- THIS sorts the bars
            color="#FF8C00"
        )

        plt.title(f"Countplot of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Number of Users")
        plt.xticks(rotation=45)
        plt.tight_layout()

        fig = plt.gcf()
        return fig
    return (countplot_seaborn,)


@app.cell
def _(np, plt):
    def scatter_matplotlib(df_users_3 , x_col, y_col):
        plt.figure(figsize=(6, 4))

        # Unique clusters
        clusters = df_users_3["cluster_behavior"].unique()

        # Generate orange shades
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(clusters)))

        # Plot each cluster
        for cluster, color in zip(clusters, colors):
            subset = df_users_3[df_users_3["cluster_behavior"] == cluster]
            plt.scatter(
                subset[x_col],
                subset[y_col],
                color=color,
                alpha=0.8,
                label=str(cluster),)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
        plt.legend(title="cluster_behavior")
        plt.tight_layout()

        return plt.gcf()
    return (scatter_matplotlib,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Getting Our Data
    """) if mo.app_meta().mode == "edit" else None
    return


@app.cell
def _():
    url="https://raw.githubusercontent.com/FrankCP/ML_Team7/main/data/"
    return (url,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visual Looking To Our Datasets
    """) if mo.app_meta().mode == "edit" else None
    return


@app.cell
def _(pd, url):
    df_calendar = pd.read_csv(url+'calendar.csv')
    return (df_calendar,)


@app.cell
def _(pd, url):
    df_city = pd.read_csv(url+'city.csv')
    return (df_city,)


@app.cell
def _(pd, url):
    df_passenger= pd.read_csv(url+'passenger.csv')
    return (df_passenger,)


@app.cell
def _(pd, url):
    df_trip = pd.read_csv(url+'trip.csv.gz')
    return (df_trip,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Preparation: Cleaning & Preprocessing
    """) if mo.app_meta().mode == "edit" else None
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
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > As we see, the time columns are strings instead of datetime
    """) if mo.app_meta().mode == "edit" else None
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
    #df_users_1['Churn'] = df_users['Churn'] 
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
    """) if mo.app_meta().mode == "edit" else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature Engineering Part
    """) if mo.app_meta().mode == "edit" else None
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
    #df_users_1
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
    """) if mo.app_meta().mode == "edit" else None
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
    """) if mo.app_meta().mode == "edit" else None
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
def _(
    KMeans,
    X_scaled,
    cluster_features,
    df_users_2,
    n_clusters_slider,
    pd,
    scaler,
):
    best_k = n_clusters_slider.value
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    _cluster_labels = kmeans_final.fit_predict(X_scaled)



    # Let's get the centers and figure out the charaterics of our clusters
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans_final.cluster_centers_),
        columns=cluster_features.columns
    )

    # Compute score for ordering clusters
    centers["score"] = (
        centers["Customer_Lifetime_Days"].rank(ascending=True) +
        centers["total_trip_fare"].rank(ascending=True)
    )


    # Order clusters by score (highest first)
    ordered_clusters = centers["score"].sort_values(ascending=False).index.tolist()


    # Create alphabetical labels
    labels = [chr(65 + i) for i in range(len(ordered_clusters))]

    # Create map: cluster index → letter label
    cluster_label_map = dict(zip(ordered_clusters, labels))

    # Apply mapping directly to the predicted labels
    _cluster_labels = pd.Series(_cluster_labels).map(cluster_label_map).values


    df_users_2['cluster_behavior'] = _cluster_labels
    return (centers,)


@app.cell
def _(df_city, df_users_2):
    #centers
    df_users_2['first_city'] = df_users_2['first_city'].map(
        df_city.set_index('city_id')['city_name']
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 99 Little Orange - Dashboard
    """)
    return


@app.cell
def _(mo):
    churn_threshold_slider = mo.ui.slider(
        start=28, stop=180, value=28,
        label="Churn inactivity threshold (days)"
    )
    return (churn_threshold_slider,)


@app.cell
def _(mo):
    n_clusters_slider = mo.ui.slider(start=3, stop=12, step=1, label="Number of clusters")
    return (n_clusters_slider,)


@app.cell
def _(df_users_2, mo):
    multi_cluster= mo.ui.multiselect(
        options=df_users_2['cluster_behavior'].unique(),
        label="Choose Cluster Category Or More",
        value=[],      
        )
    return (multi_cluster,)


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
    multi_cluster,
    n_clusters_slider,
):
    mo.vstack([
        churn_threshold_slider,
        n_clusters_slider,
        multi_cluster,
        mo.hstack([date_from, date_to, apply_button])
    ])
    return


@app.cell
def _(date_from, date_to, df_users, df_users_2, multi_cluster):
    df_users_3 = df_users_2[(df_users_2['last_trip_date'].between(str(date_from.value),str(date_to.value)))|(df_users_2['cluster_behavior'].isin(multi_cluster.value))].merge(df_users[['passenger_id', 'Churn']], on ='passenger_id', how= 'left')
    return (df_users_3,)


@app.cell
def _(df_users_3, mo):
    cols = df_users_3.select_dtypes("number").columns.tolist()
    col_cat = (
        df_users_3
        .drop(columns="passenger_id", errors="ignore")
        .select_dtypes("object")
        .columns
        .tolist()
    )

    feature_for_bar = mo.ui.dropdown(
        options=col_cat,
        label="Choose a feature for the barplot",
        value=col_cat[0]
    )

    feature_select_x = mo.ui.dropdown(
        options=cols,
        label="Choose x feature for the barplot",
        value=cols[0]
    )

    feature_select_y = mo.ui.dropdown(
        options=cols,
        label="Choose y feature for the barplot",
        value=cols[0]
    )
    return feature_for_bar, feature_select_x, feature_select_y


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
    return card1, card2, card3, card4


@app.cell
def _(card1, card2, card3, card4, mo):
    # Display them beside each other (like Power BI)
    mo.hstack([card1, card2, card3, card4], widths="equal", gap=2)
    return


@app.cell
def _(centers, df_users_2, np, plt):
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
    _cluster_labels = df_users_2['cluster_behavior'].unique()
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
    #plt.show()
    plt.gcf()
    return


@app.cell
def _(feature_for_bar):
    feature_for_bar
    return


@app.cell
def _(countplot_seaborn, df_users_3, feature_for_bar):
    countplot_seaborn(df_users_3, feature_for_bar.value)
    return


@app.cell
def _(feature_select_x, feature_select_y, mo):
    mo.hstack([feature_select_x, feature_select_y])
    return


@app.cell
def _(df_users_3, feature_select_x, feature_select_y, scatter_matplotlib):

    scatter_matplotlib(df_users_3, feature_select_x.value, feature_select_y.value)
    return


@app.cell
def _(df_users_3, mo):
    multi_features = mo.ui.multiselect(
        options=df_users_3.columns.tolist(),
        label="Choose multiple columns",
        value=[],      
        )


    mo.vstack([multi_features])
    return (multi_features,)


@app.cell
def _(df_users_3):
    df_users_3['cluster_behavior'].unique()
    return


@app.cell
def _(df_users_3, mo, multi_features):
    mo.ui.table(df_users_3[multi_features.value])
    return


if __name__ == "__main__":
    app.run()
