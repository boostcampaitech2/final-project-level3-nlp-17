column_dict = {
    "bluetopgp": [],
    "bluetopwr": [],
    "bluetopchwr": [],
    "bluetopkda": [],
    "bluejunglegp": [],
    "bluejunglewr": [],
    "bluejunglechwr": [],
    "bluejunglekda": [],
    "bluemidgp": [],
    "bluemidwr": [],
    "bluemidchwr": [],
    "bluemidkda": [],
    "blueadcgp": [],
    "blueadcwr": [],
    "blueadcchwr": [],
    "blueadckda": [],
    "bluesupportgp": [],
    "bluesupportwr": [],
    "bluesupportchwr": [],
    "bluesupportkda": [],
    "redtopgp": [],
    "redtopwr": [],
    "redtopchwr": [],
    "redtopkda": [],
    "redjunglegp": [],
    "redjunglewr": [],
    "redjunglechwr": [],
    "redjunglekda": [],
    "redmidgp": [],
    "redmidwr": [],
    "redmidchwr": [],
    "redmidkda": [],
    "redadcgp": [],
    "redadcwr": [],
    "redadcchwr": [],
    "redadckda": [],
    "redsupportgp": [],
    "redsupportwr": [],
    "redsupportchwr": [],
    "redsupportkda": [],
    "win": list(map(int, data["win"])),
}


def separate_columns(index, post_fix, column_name):
    match = data.iloc[index]
    tmp_key = [key for key in column_dict.keys() if key[-len(post_fix) :] == post_fix]
    tmp_col = match[column_name]
    for key, col in zip(tmp_key, tmp_col):
        column_dict[key].append(col)


for i in tqdm(data.index):
    separate_columns(i, "gp", "participants_and_champions_played")
    separate_columns(i, "wr", "participants_and_champions_win_rates")
    separate_columns(i, "chwr", "champions_win_rates")
    separate_columns(i, "kda", "kda")

data_preprocessed = pd.DataFrame(column_dict)
data_preprocessed
