import numpy as np
import pandas as pd


# recommender class
class pop_recommender():
    def __init__(self):
        self.training_data = None
        self.user_id = None
        self.item_id = None
        self.recommendations = None

    def create(self, train_data, user_id, item_id):
        self.training_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # group by item_id (in this case it will be song), and count the number of users that listened to it
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={'user_id': 'score'}, inplace=True)

        # sort songs on score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0, 1])

        # put a rank to each song
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        #get top 100
        self.popularity_recommendations = train_data_sort.head(100)

    # make recs
    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        # put the user id in the list
        user_recommendations['user_id'] = user_id

        # Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations



# Class for Item similarity based Recommender System model
class item_recommender():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.co_occurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None

    # gets unique songs for a user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        return list(user_data[self.item_id].unique())

    # gets users that listened to a song
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        #sets dont have dups
        return set(item_data[self.user_id].unique())

    # gets all songs
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())

        return all_items

    # makes co-occurence matrix
    def make_co_occurence_matrix(self, user_songs, all_songs):

        #get the users songs
        user_songs_users = []
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        #initialize the matrix
        co_occurence_matrix = np.zeros(shape=(len(user_songs), len(all_songs)))


        for i in range(0, len(all_songs)):
            #get unique users of a song,i
            songs_data_i = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_data_i[self.user_id].unique())

            for j in range(0, len(user_songs)):

                #get unique users of a song,i
                users_j = user_songs_users[j]

                #intersection of users who listened to songs i and j
                users_intersection = users_i.intersection(users_j)

                # Calculate co-occurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    # Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)

                    co_occurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    co_occurence_matrix[j, i] = 0

        return co_occurence_matrix

    def generate_top_recommendations(self, user, co_occurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(co_occurence_matrix))

        # get weighted averages
        user_scores = co_occurence_matrix.sum(axis=0) / float(co_occurence_matrix.shape[0])
        user_scores = np.array(user_scores)[0].tolist()

        # sort on value of user_scores
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_scores))), reverse=True)

        df = pd.DataFrame(columns=['user_id', 'song', 'score', 'rank'])

        #populate df with top 100 item based recommendations
        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)] = [user, all_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        # if no recs
        if df.shape[0] == 0:
            print("This user does not have enough data collected to generate recommendations.")
            return -1

        return df

    # creator function
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    # make recs
    def recommend(self, user):

        #get all unique songs for the user
        user_songs = self.get_user_items(user)

        print("The user listened to : %d" % len(user_songs))

        #get all the songs from set
        all_songs = self.get_all_items_train_data()

        print("no. of unique songs in the training set: %d" % len(all_songs))

        #make co-occurence matrix
        co_occurence_matrix = self.make_co_occurence_matrix(user_songs, all_songs)

        #make recs
        df_recommendations = self.generate_top_recommendations(user, co_occurence_matrix, all_songs, user_songs)

        return df_recommendations

    # get similar songs to the input song
    def get_similar_items(self, item_list):

        user_songs = item_list

        #get all songs
        all_songs = self.get_all_items_train_data()

        #make co-occurence matrix
        co_occurence_matrix = self.make_co_occurence_matrix(user_songs, all_songs)

        #get recs
        user = ""
        df_recommendations = self.generate_top_recommendations(user, co_occurence_matrix, all_songs, user_songs)

        return df_recommendations