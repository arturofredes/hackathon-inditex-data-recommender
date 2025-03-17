import pandas as pd
import datetime as dt

def get_session_metrics(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Given a pandas DataFrame in the format of the train dataset and a user_id, return the following metrics for every session_id of the user:
        - user_id (int) : the given user id.
        - session_id (int) : the session id.
        - total_session_time (float) : The time passed between the first and last interactions, in seconds. Rounded to the 2nd decimal.
        - cart_addition_ratio (float) : Percentage of the added products out of the total products interacted with. Rounded ot the 2nd decimal.

    If there's no data for the given user, return an empty Dataframe preserving the expected columns.
    The column order and types must be scrictly followed.

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    user_id : int
        Id of the client.

    Returns
    -------
    Pandas Dataframe with some metrics for all the sessions of the given user.
    """
    user = df[df['user_id']==user_id]
    #If there's no data for the given user, return an empty Dataframe preserving the expected columns.
    if user.shape[0] == 0:
        return pd.DataFrame(columns = ["user_id","session_id", "total_session_time", "cart_addition_ratio"])
    
    user['timestamp_local'] = pd.to_datetime(user['timestamp_local'])
    metrics = user.groupby(['user_id', 'session_id']).agg(
        min_timestamp=('timestamp_local', 'min'),
        max_timestamp=('timestamp_local', 'max'),
        partnumber_count=('partnumber', 'count'),
        add_to_cart_sum=('add_to_cart', 'sum')
    ).reset_index()

    metrics["total_session_time"] = (
    (metrics["max_timestamp"] - metrics["min_timestamp"]).dt.total_seconds().round(2)
    )
    metrics["cart_addition_ratio"] = ((metrics["add_to_cart_sum"] / metrics["partnumber_count"])*100).round(2)

    #sorted in ascending order by user identifier and then by session identifier
    metrics.sort_values(by=["user_id","session_id"], ascending=[True, True], inplace=True)


    return metrics[["user_id","session_id", "total_session_time", "cart_addition_ratio"]]