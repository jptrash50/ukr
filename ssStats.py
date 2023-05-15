##########
# z-score item normalization
# ddof = Means Delta Degrees of Freedom (default = 0)
#        The divisor used in Standard Deviation calculations is (N - ddof),
#        where N represents the number of elements.
#
# https://stackoverflow.com/questions/23451244/how-to-zscore-normalize-pandas-column-with-nans
#####
def zscore(df, ddof=0):
    return (df - df.mean()) / df.std(ddof=ddof)


##########
# Cronbach Alpha scale reliability / internal consistency
# >=0.9 Excellent, >=0.8 Good, >=0.7 Acceptable, >=0.6 Questionable, >=0.5 Poor, <0.5 Unacceptable
#
# https://stackoverflow.com/questions/20799403/improving-performance-of-cronbach-alpha-code-python-numpy
#####
def alpha(df):
    # Ensure the dataframe has no missing items
    df = df.dropna()

    # Number of columns/items, aka: len(df.columns)
    num_items = df.shape[1]

    # If there are not at least two items in the scale, then return a 1
    if num_items < 2:
        return 1

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.var.html
    sum_of_item_variances = df.var(axis=0).sum()
    variance_of_sum_of_items = df.sum(axis=1).var()

    # Return the calculated alpha value
    return num_items / (num_items - 1) * (1 - sum_of_item_variances / variance_of_sum_of_items)


##########
# Cronbach Alpha Best
# From the list of all cronbach alphas, return the potentially best option
# min_best = Minimum acceptable alpha (default = 0.7)
# drop_penalty = an alpha penalty for every dropped item (default 1%)
#####
def alpha_best(alphas, min_best=0.7, drop_penalty=0.01):
    # Ensure the alpha-list is sorted by fewest drops then alpha
    # [flag, alpha, included count, included columns, drop count, dropped columns]
    sorted(alphas, key=lambda _: (-_[2], -_[1]))

    # Set the initial best value as the top item (no drops)
    best = alphas[0]

    for _ in alphas:
        (flag, alpha, icount, included, dcount, dropped) = _

        # Create a penalty against the current alpha for every extra drop it has over the best
        penalty = 1 - (dcount - best[4]) * drop_penalty

        # If current alpha (times any drop penalty) is better than selected best, then update the best to current
        if alpha * penalty > best[1]:
            best = _

        # If the selected best is acceptable, then we're done
        if best[1] >= min_best:
            break

    return best


##########
# Cronbach Alpha drops
# Recursively check alpha results dropping various items in the scale
#####
def alpha_drops(df, dcount=0, dropped=None, position=0, max_drops=10, min_items=2, rtrn=None):
    if min_items < 2:
        min_items = 2

    if dropped is None:
        dropped = []

    if rtrn is None:
        rtrn = []

    from warnings import filterwarnings
    filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

    # Output the Cronbach Alpha for this scale, flagging questionable, acceptable, good, excellent results
    ca = alpha(df)
    flag = '*' * max(0, int(ca * 10) - 5)
    # [flag, alpha, included count, included columns, drop count, dropped columns]
    rtrn.append([flag, ca, len(df.columns), list(df.columns), dcount, dropped])

    # Check that there are more than the minimum number of columns remaining
    # and that no more than the max number of drops have already been done
    if len(df.columns) > min_items and dcount < max_drops:
        # Drop another item from the scale and check again
        for column in df.columns[position:]:
            position = df.columns.get_loc(column)
            _df = df.drop(column, axis=1)
            rtrn = alpha_drops(_df, dcount + 1, dropped + [column], position, max_drops, min_items, rtrn)

    return sorted(rtrn, key=lambda _: (-_[2], -_[1]))
