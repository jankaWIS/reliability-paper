import os

import numpy as np
import pandas as pd


def process_and_concatenate_all_tasks(task_names, task_files, path_results):

    ## Define all variables

    # make a copy, should be enough since the objects are strings
    check_used_tasks = task_files[:]

    # list of dataframes
    dataframe_list = []

    # dic of num of subjects per task
    dic_num_subjects_per_task = {}

    # list of id of subjects who have everything
    completed_subjects = []

    # list of id of subjects who have anything
    all_anything_subjects = []

    # list of id of subjects who have at least one repetition of all tasks
    one_repeat_all_tasks_subjects = []

    # list of id of subjects who have all three VET tasks
    VET_shared_subjects = []

    # wide format
    df_wide = pd.DataFrame(columns=['userID'])
    # for mean scores that are normed to be within 0 and 1
    df_normed_wide = pd.DataFrame(columns=['userID'])

    # Process and load
    for t in task_names:
        print(t)
        first_batch = {}

        # define what to use
        if t == "MST":
            usecols = ['userID', 'correct', 'trial_type', 'repetition', 'response']
        elif t == "PGNG":
            usecols = ['userID', 'correct', 'block', 'inhibitory']
        elif t == "nback":
            usecols = ["userID", "level", "correct_response", "response_type", "correct"]
        elif t == "PIM_MC":
            usecols = ["userID", "hobby", "vice", "code_name", "country"]
        elif t == "SCAP":
            usecols = ['userID', 'correct', 'set_size']
        else:
            usecols = ['userID', 'correct']

        # go over special cases
        if t == "CFMT":
            # load participants for the original (first) form
            first_batch = set(pd.read_csv(os.path.join(path_results, "CFMT-cleaned_data.csv"), usecols=['userID']).userID)

            # load participants per each of the 4 forms
            f2 = set(pd.read_csv(os.path.join(path_results, "CFMT_F2-cleaned_data.csv"), usecols=['userID']).userID)
            f = set(pd.read_csv(os.path.join(path_results, "CFMT_F-cleaned_data.csv"), usecols=['userID']).userID)
            aus2 = set(pd.read_csv(os.path.join(path_results, "CFMT_Aus2-cleaned_data.csv"), usecols=['userID']).userID)
            aus = set(pd.read_csv(os.path.join(path_results, "CFMT_Aus-cleaned_data.csv"), usecols=['userID']).userID)

            # select participants from the second who do not have the first to include them
            take_aus2 = set(aus2) - set(aus)  # take from aus 2 everyone who does not have aus 1
            take_f2 = set(f2) - set(f)  # take from female 2 everyone who does not have female 1

            # get participants who have all data
            shared_participants = list(first_batch & (aus|aus2) & (f|f2))
            print(f"Running total {len(shared_participants)} participants (out of {len(first_batch)} in first repetition).")

            # load and concat the datasets, condition the loading on the given subjects, see
            # https://stackoverflow.com/questions/28239529/conditional-row-read-of-csv-in-pandas
            # NOTE -- you have to AND the shared participants and the ones that you want to take from second Aus and F
            # Add a separate column specifying the version loaded
            df_concat = pd.concat([
                (pd.read_csv(os.path.join(path_results, 'CFMT-cleaned_data.csv'))[lambda x: x['userID'].isin(shared_participants)]).assign(form='original'),
                (pd.read_csv(os.path.join(path_results, 'CFMT_Aus2-cleaned_data.csv'))[lambda x: x['userID'].isin(set(shared_participants) & take_aus2)]).assign(form='Aus2'),
                (pd.read_csv(os.path.join(path_results, 'CFMT_F2-cleaned_data.csv'))[lambda x: x['userID'].isin(set(shared_participants) & take_f2)]).assign(form='F2'),
                (pd.read_csv(os.path.join(path_results, 'CFMT_Aus-cleaned_data.csv'))[lambda x: x['userID'].isin(shared_participants)]).assign(form='Aus1'),
                (pd.read_csv(os.path.join(path_results, 'CFMT_F-cleaned_data.csv'))[lambda x: x['userID'].isin(shared_participants)]).assign(form='F1'),
            ])

            # update list
            check_used_tasks.remove("CFMT-cleaned_data.csv")
            check_used_tasks.remove("CFMT_Aus-cleaned_data.csv")
            check_used_tasks.remove("CFMT_F-cleaned_data.csv")
            check_used_tasks.remove("CFMT_Aus2-cleaned_data.csv")
            check_used_tasks.remove("CFMT_F2-cleaned_data.csv")

        elif t == "emotion_matching":
            # load the first set
            df_emotion_matching = pd.read_csv(os.path.join(path_results, "emotion_matching-cleaned_data.csv"))
            df_emotion_matching["form"] = "original"
            df_emotion_matching["correct"] = df_emotion_matching["correct"].astype(float)

            # should be empty -- test if people don't have more than 1 entry
            print("Test first set: ", t)
            print(df_emotion_matching.userID.value_counts()[df_emotion_matching.userID.value_counts() != 65])  # .keys()

            # load the second set
            df_emotion_matching_stand = pd.read_csv(os.path.join(path_results, "emotion_matching_rep-cleaned_data.csv"))
            df_emotion_matching_stand["form"] = "repetition"
            df_emotion_matching_stand["correct"] = df_emotion_matching_stand["correct"].astype(float)

            # should be empty -- test if people don't have more than 1 entry
            print("\nTest second set")
            print(df_emotion_matching_stand.userID.value_counts()[
                      df_emotion_matching_stand.userID.value_counts() != 100])  # .keys()

            ######

            # get names of stimuli
            df_emotion_matching_stand["face1"] = df_emotion_matching_stand["face1"].str.split('/').str[-1]
            df_emotion_matching_stand["face2"] = df_emotion_matching_stand["face2"].str.split('/').str[-1]
            df_emotion_matching_stand["face3"] = df_emotion_matching_stand["face3"].str.split('/').str[-1]

            # to be able to take the old trials only, here, I create a new variable "test" which is a combination of
            # the 3 stimuli which uniquely determine a trial. Since the assignment of the stimuli to a column is
            # random, we can simply compare. Therefore, I first combine the three cols, then sort values within this
            # and convert it into a string. Using this string and userID, I then merge the two datasets --> that
            # means that there are only the shared participants together with the overlapping trials
            df_emotion_matching_stand["test"] = np.sort(df_emotion_matching_stand[["face1", "face2", "face3"]].values, axis=1).tolist()
            df_emotion_matching["test"] = np.sort(df_emotion_matching[["face1", "face2", "face3"]].values, axis=1).tolist()

            df_emotion_matching["test"] = df_emotion_matching["test"].apply(lambda x: ','.join(map(str, x)))
            df_emotion_matching_stand["test"] = df_emotion_matching_stand["test"].apply(lambda x: ','.join(map(str, x)))

            # merge the two dataframes to only take the intersection -- takes care of both userID and the shared target
            df_merged = df_emotion_matching.merge(df_emotion_matching_stand, on=["userID", "test"])

            # split it and make it into a long format, back, rename cols to be the same again
            df_concat = pd.concat([
                df_merged.loc[:, [x for x in df_merged.columns if x.endswith('_x') or x == 'userID']].rename(columns=lambda x: x.strip('_x')),
                df_merged.loc[:, [x for x in df_merged.columns if x.endswith('_y') or x == 'userID']].rename(columns=lambda x: x.strip('_y')),
            ])

            # get participants on the first visit
            first_batch = set(df_emotion_matching['userID'])
            # get participants who have both data
            shared_participants = list(first_batch & set(df_emotion_matching_stand['userID']))
            print(f"Running total {len(shared_participants)} participants (out of {len(first_batch)} in first repetition).")

            # update list
            check_used_tasks.remove("emotion_matching-cleaned_data.csv")
            check_used_tasks.remove("emotion_matching_rep-cleaned_data.csv")

        elif t == "emotion_labelling":
            # load the first set
            df_emotion_labelling = pd.read_csv(os.path.join(path_results, "emotion_labelling-cleaned_data.csv"))
            df_emotion_labelling["form"] = "original"
            df_emotion_labelling["correct"] = df_emotion_labelling["correct"].astype(float)

            # should be empty -- test if people don't have more than 1 entry
            print("Test first set: ", t)
            print(df_emotion_labelling.userID.value_counts()[df_emotion_labelling.userID.value_counts() != 48])  #.keys()

            df_emotion_labelling_stand = pd.read_csv(os.path.join(path_results, "emotion_labelling_rep-cleaned_data.csv"))
            df_emotion_labelling_stand["form"] = "repetition"
            df_emotion_labelling_stand["correct"] = df_emotion_labelling_stand["correct"].astype(float)

            # should be empty -- test if people don't have more than 1 entry
            print("\nTest second set")
            print(df_emotion_labelling_stand.userID.value_counts()[df_emotion_labelling_stand.userID.value_counts()>100])  #.keys()

            ######

            df_emotion_labelling_stand["target"] = df_emotion_labelling_stand["target"].str.split('/').str[-1]

            # get participants on the first visit
            first_batch = set(df_emotion_labelling['userID'])
            # get participants who have both data
            shared_participants = list(first_batch & set(df_emotion_labelling_stand['userID']))

            # get target set of original stimuli
            shared_target = list(set(df_emotion_labelling['target']) & set(df_emotion_labelling_stand['target']))
            print(f"Running all replicate studies with total {len(shared_participants)} participants (out of {len(first_batch)} in first repetition) and {len(shared_target)} stimuli.")

            df_concat = pd.concat([
                df_emotion_labelling_stand.loc[(df_emotion_labelling_stand["target"].isin(shared_target)) & \
                                               (df_emotion_labelling_stand["userID"].isin(shared_participants))].copy(),
                df_emotion_labelling.loc[(df_emotion_labelling["target"].isin(shared_target)) & \
                                         (df_emotion_labelling["userID"].isin(shared_participants))].copy()
            ])

            # update list
            check_used_tasks.remove("emotion_labelling-cleaned_data.csv")
            check_used_tasks.remove("emotion_labelling_rep-cleaned_data.csv")

        elif t == "RISE":
            # get participants on the first visit
            first_batch = set(pd.read_csv(os.path.join(path_results, "RISE-cleaned_data.csv"), usecols=['userID']).userID)
            # get participants who have both data
            shared_participants = list(
                 first_batch #&\
                 # set(pd.read_csv(os.path.join(path_results, "RISE-cleaned_data_rep.csv"), usecols=['userID']).userID)
            )
            print(f"Running total {len(shared_participants)} participants (out of {len(first_batch)} in first repetition).")

            # load and concat the datasets, add label to know the form
            df_concat = pd.concat([
                pd.read_csv(os.path.join(path_results, 'RISE-cleaned_data.csv')).assign(form='original'),
                # pd.read_csv(os.path.join(path_results, 'RISE-cleaned_data_rep.csv')),
            ])

            # rename cols to take the correct answer
            df_concat.rename(columns={"correct": "for_sure_wrong_correct", "my_correct": "correct"}, inplace=True)
            # update list
            check_used_tasks.remove("RISE-cleaned_data.csv")

        elif t == "nback":
            # get participants on the first visit
            first_batch = set(pd.read_csv(os.path.join(path_results, "nback-cleaned_data_day1.csv"), usecols=['userID']).userID)
            # get participants who have both data
            shared_participants = list(
                first_batch & \
                set(pd.read_csv(os.path.join(path_results, "nback-cleaned_data_day2.csv"), usecols=['userID']).userID)
            )
            print(f"Running total {len(shared_participants)} participants (out of {len(first_batch)} in first repetition).")

            # load and concat the datasets
            df_concat = pd.concat([
                pd.read_csv(os.path.join(path_results, 'nback-cleaned_data_day1.csv'), usecols=usecols).assign(form='original'),
                pd.read_csv(os.path.join(path_results, 'nback-cleaned_data_day2.csv'), usecols=usecols).assign(form='repetition'),
            ])

            # update list
            check_used_tasks.remove("nback-cleaned_data_day1.csv")
            check_used_tasks.remove("nback-cleaned_data_day2.csv")

        elif t == "Navon":
            # get participants on the first visit
            first_batch = set(pd.read_csv(os.path.join(path_results, "Navon-cleaned_data.csv"), usecols=['userID']).userID)
            # get participants who have both data
            shared_participants = list(
                first_batch  # &\
            )
            print(f"Running total {len(shared_participants)} participants (out of {len(first_batch)} in first repetition).")

            # load and concat the datasets
            df_concat = pd.concat([
                pd.read_csv(os.path.join(path_results, 'Navon-cleaned_data.csv'), usecols=["userID", "correct_flt"]).assign(form='original'),
            ])
            # rename col
            df_concat.rename(columns={"correct_flt": "correct"}, inplace=True)

            # update list
            check_used_tasks.remove("Navon-cleaned_data.csv")

        elif t.startswith("VET"):
            # get participants on the first visit
            first_batch = set(pd.read_csv(os.path.join(path_results, f"{t}-cleaned_data.csv"), usecols=['userID']).userID)
            # get participants who have both data
            shared_participants = list(first_batch)
            print(f"Running total {len(shared_participants)} participants (out of {len(first_batch)} in first repetition).")

            # load and concat the datasets
            df_concat = pd.read_csv(os.path.join(path_results, f'{t}-cleaned_data.csv')).assign(form=t)

            # update list
            check_used_tasks.remove(f"{t}-cleaned_data.csv")

        else:
            # get participants on the first visit
            first_batch = set(pd.read_csv(os.path.join(path_results, f"{t}-cleaned_data.csv"), usecols=['userID']).userID)
            # get participants who have both data
            shared_participants = list(
                first_batch & \
                set(pd.read_csv(os.path.join(path_results, f"{t}-cleaned_data_rep.csv"), usecols=['userID']).userID)
            )
            print(f"Running total {len(shared_participants)} participants (out of {len(first_batch)} in first repetition).")

            if t == "FMP":
                # load and concat the datasets
                df_concat = pd.concat([
                    pd.read_csv(os.path.join(path_results, f'FMP-cleaned_data.csv'), usecols=["userID", "task", "difficulty", "correct_flt"]).assign(form='original'),
                    pd.read_csv(os.path.join(path_results, f'FMP-cleaned_data_rep.csv'), usecols=["userID", "task", "difficulty", "correct_flt"]).assign(form='repetition'),
                ])
                # rename column to be consistent
                df_concat.rename(columns={"correct_flt": "correct", "task": "memory_load"}, inplace=True)

                # update usecols
                usecols += ['difficulty', 'memory_load']

            elif t == "GFMT":
                # get overlaping stimuli
                shared_target = list(
                    set(pd.read_csv(os.path.join(path_results, "GFMT-cleaned_data.csv"), usecols=['FaceStim']).FaceStim) & \
                    set(pd.read_csv(os.path.join(path_results, "GFMT-cleaned_data_rep.csv"), usecols=['FaceStim']).FaceStim)
                )

                # load and concat the datasets
                df_concat = pd.concat([
                    pd.read_csv(os.path.join(path_results, 'GFMT-cleaned_data.csv')).assign(form='original'),
                    pd.read_csv(os.path.join(path_results, 'GFMT-cleaned_data_rep.csv')).assign(form='repetition'),
                ])

                # take only the full data
                df_concat = df_concat[(df_concat["FaceStim"].isin(shared_target)) & (
                    df_concat["userID"].isin(shared_participants))].reset_index(drop=True)
                # take only columns of interest
                df_concat = df_concat[usecols+["form"]].copy()

            elif t == "PGNG":
                # load and concat the datasets
                df_concat = pd.concat([
                    pd.read_csv(os.path.join(path_results, 'PGNG-cleaned_data.csv')).assign(form='original'),
                    pd.read_csv(os.path.join(path_results, 'PGNG-cleaned_data_rep.csv')).assign(form='repetition'),
                ])
                # drop the first block
                df_concat = df_concat[df_concat["block"] != 1]
                # take only columns of interest
                df_concat = df_concat[usecols+["form"]].copy()

            else:
                # load and concat the datasets
                df_concat = pd.concat([
                    pd.read_csv(os.path.join(path_results, f'{t}-cleaned_data.csv'), usecols=usecols).assign(form='original'),
                    pd.read_csv(os.path.join(path_results, f'{t}-cleaned_data_rep.csv'), usecols=usecols).assign(form='repetition'),
                ])

            # update done tasks
            check_used_tasks.remove(f"{t}-cleaned_data.csv")
            check_used_tasks.remove(f"{t}-cleaned_data_rep.csv")

        ### Process
        # add form to usecols
        usecols += ["form"]
        # take only the full data
        df_concat = df_concat.loc[df_concat["userID"].isin(shared_participants), usecols].reset_index(drop=True)
        # add a label
        df_concat["task"] = t

        if t == "PIM_MC":

            df_wide = pd.merge(df_wide, df_concat.groupby(['userID']).sum().reset_index(), on="userID", how='outer')

            # add the overall acc
            y = pd.melt(df_concat[['userID', 'hobby', 'vice', 'code_name', 'country']], id_vars='userID',
                        value_name='correct').groupby(['userID']).sum().reset_index()
            # do the wide
            df_wide = pd.merge(df_wide, y.rename(columns={'correct': t}), on="userID", how='outer')
            y = None

            ## add mean accuracy
            df_normed_wide = pd.merge(df_normed_wide, df_concat.groupby(['userID']).mean().reset_index(), on="userID", how='outer')
            # add the overall acc
            y = pd.melt(df_concat[['userID', 'hobby', 'vice', 'code_name', 'country']], id_vars='userID',
                        value_name='correct').groupby(['userID']).mean().reset_index()
            # do the normed_wide
            df_normed_wide = pd.merge(df_normed_wide, y.rename(columns={'correct': t}), on="userID", how='outer')
            y = None

        else:
            # add results per subject
            tmp = df_concat.convert_dtypes().groupby(["userID"]).sum().correct.reset_index()
            # first do the wide
            if df_wide.empty:
                df_wide = tmp.rename(columns={'correct': t})
            else:
                df_wide = pd.merge(df_wide, tmp.rename(columns={'correct': t}), on="userID", how='outer')

            if t == "FMP":
                y = df_concat.convert_dtypes().groupby(["memory_load", "userID"]).sum().reset_index()
                for tsk in ['perception', 'blank', 'faces']:
                    df_wide = pd.merge(df_wide,
                                       y.loc[y["memory_load"] == tsk, ["userID", "correct"]].rename(columns={'correct': f'{t}_{tsk}'}),
                                       on="userID", how='outer')

                    # # add pseudoslopes, it's way too many steps together. It first sums per memory condition, resets index, then does subtraction of faces-perception columns
                    # df_wide = pd.merge(df_wide,
                    #                    (df_concat.groupby(["memory_load", "userID"]).sum().correct.unstack(level=0).loc[:,["perception", "faces"]].diff(axis=1).iloc[:, -1]).reset_index().rename(columns={'faces': f'{t}_faces-perc'}),
                    #                    on="userID", how='outer')
                # alternative and much simpler way is to just subtract the columns
                df_wide["FMP_faces-perc"] = df_wide["FMP_faces"] - df_wide["FMP_perception"]

            tmp = None
            ## add mean
            tmp = df_concat.convert_dtypes().groupby(["userID"]).mean().correct.reset_index()
            # first do the normed_wide
            if df_normed_wide.empty:
                df_normed_wide = tmp.rename(columns={'correct': t})
            else:
                df_normed_wide = pd.merge(df_normed_wide, tmp.rename(columns={'correct': t}), on="userID", how='outer')

            if t=="FMP":
                y = df_concat.convert_dtypes().groupby(["memory_load", "userID"]).mean().reset_index()
                for tsk in ['perception', 'blank', 'faces']:
                    df_normed_wide = pd.merge(df_normed_wide, y.loc[y["memory_load"]==tsk,["userID","correct"]].rename(columns={'correct': f'{t}_{tsk}'}), on="userID", how='outer')

                # alternative and much simpler way is to just subtract the columns
                df_normed_wide["FMP_faces-perc"] = df_normed_wide["FMP_faces"]-df_normed_wide["FMP_perception"]

        # update dict
        dic_num_subjects_per_task[t] = len(shared_participants)

        # append df
        dataframe_list.append(df_concat)

        # update subjects' list
        if completed_subjects:
            completed_subjects = set(completed_subjects) & set(shared_participants)
        else:
            completed_subjects = shared_participants[:]

        if one_repeat_all_tasks_subjects:
            one_repeat_all_tasks_subjects = set(one_repeat_all_tasks_subjects) & set(first_batch)
        else:
            one_repeat_all_tasks_subjects = list(first_batch)[:]

        if all_anything_subjects:
            all_anything_subjects = set(all_anything_subjects) | set(shared_participants)
        else:
            all_anything_subjects = shared_participants[:]

        if t.startswith("VET"):
            if VET_shared_subjects:
                VET_shared_subjects = set(VET_shared_subjects) & set(shared_participants)
            else:
                VET_shared_subjects = shared_participants[:]

        # clean memory
        df_concat = None
        shared_participants = None

    # update VET subjects
    dic_num_subjects_per_task["VET_birds_planes_leaves"] = len(VET_shared_subjects)

    print(f'\n\nThis should be as before: {len(task_files)}; then this is how many files is left and what they are:\n',
          len(check_used_tasks), check_used_tasks)
    print(f'\n-->There are {len(completed_subjects)} subjects who have all the tasks.')
    print(f'\n-->There are {len(all_anything_subjects)} subjects who have all repetitions of at least one task.')
    print(f'\n-->There are {len(one_repeat_all_tasks_subjects)} subjects who have at least one repetition of all tasks.')

    return dataframe_list, dic_num_subjects_per_task, df_wide, df_normed_wide, completed_subjects, all_anything_subjects, VET_shared_subjects


def fill_in_userID(df, output=False):
    """
    fill in userID everywhere based on observation, see FMP processing for more
    df: dataFrame
    output: bool, default False, if to return the df and the list
    """

    # look at all unique observations (regs), take the pairs observation:userID, shorten the df and drop NaNs
    temp = df.loc[df.observation.isin(df.observation.unique()), ["observation", "userID"]].drop_duplicates().dropna()
    # this is a conversion of the df into a dictionary, it goes through a list and then selects only the dict
    # inspiration: https://stackoverflow.com/questions/26716616/convert-a-pandas-dataframe-to-a-dictionary
    fill_userID = temp.set_index("observation").T.to_dict('records')[0]
    # now fill all the missed userIDs based on the corresponding observation value
    df.userID = df["userID"].fillna(df["observation"].map(fill_userID))

    if output:
        return df, fill_userID