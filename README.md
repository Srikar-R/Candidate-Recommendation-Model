# Candidate-Recommendation-Model
Going through resumes is hard but what if we could evaluate it without looking at it? Here is a model that for you

HireAnalytic.com  is a data science job portal that I had created so that people can find data science jobs all around the world. I had to shut it down because it required a lot of effort to maintain it but here is how I made a candidate recommending system for my job portal.


Now, while researching ideas to stand out as a job portal, I thought I could provide recruiters a special service where recruiters needn't manually search for the best candidate which can save a lot of their time and effort.


To do so, I built a candidate recommending system using K-means clustering .There are several ways to go about it, but the problem arises when you have to come up with your own score and metrics that decides what the score of an ideal candidate looks like. Therefore, I chose the K-means method which doesn't require coming up with your own values and just groups candidates with similar characteristics which looks like this:








Now, before that I had to build a resume scanner which parses the resume and extracts important information. You can read more on that here: Resume Parser 


After parsing the resume and extracting important keywords, experiences and the location of the person, the words are matched with the job description. The resume with the highest match obviously gets the highest 'resume-score', a variable which we will be using in our K-Means model.


The Dataset:

The dataset consists of people's information (name and values changed) which includes the following variables:


Name

Experience

Location ( distance from the job location)

Resume Score ( score derived from parsing the resume)

Skills percentage ( % of candidate's skills matching the skills required for the job)



Reading the dataset in python:

df=pd.read_excel("Candidate Pseudo Data.xlsx")
df.head()





Checking for outliers

We check for outliers since K-means models are very sensitive to them

plt.suptitle("Checking for outliers",color='Purple')

plt.subplot(2,2,1)
df['Experience'].plot(kind='box')

plt.subplot(2,2,2)
df['Location'].plot(kind='box')

plt.subplot(2,2,3)
df['Skills (Perc)'].plot(kind='box')

plt.subplot(2,2,4)                        
df['Resume Score'].plot(kind='box')

plt.show()



We observe that there are no outliers in the dataset and we can proceed further to standardize our model


Standardizing the dataset

Since the dataset contains variables such as experience and location which is expressed as whole numbers and Skills Perc and Resume Score which are in percentage, it is important to convert them to a common scale. Hence, we normalize it using this formula.





With python, we don't have to manually calculate it and can just use libraries to make the process faster

from sklearn.preprocessing import StandardScaler

#Creating the scaling function
scaler=StandardScaler()

#Scaling the numeric variables
num=df[['Experience', 'Location', 'Skills (Perc)', 'Resume Score']] 

#Creating a new dataset with the scaled values
df1=scaler.fit_transform(num)
df1=pd.DataFrame(df1)

#Renaming the columns
df1=df1.rename({0:'Experience',1:'Location',2:'Skills',3:'ResumeScore'},axis=1)
df1['Name']=df['Name']

df1.head()





Now, we get a dataset which is standardized. I have removed the names for now but we can always add that later.


Making the model

We use the KMeans library from sklearn module to make our model. We usually want more than 1 label or cluster for grouping. Hence, we iterate through various number of clusters and find the perfect number of clusters for our model.




# silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
# intialise kmeans
   kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
   kmeans.fit(df1)
 
   cluster_labels = kmeans.labels_
 
#silhouette score
   silhouette_avg = silhouette_score(df1, cluster_labels)
   print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

  



 Generally, we pick the values with the highest silhouette score or the one that is closer to 1. Although 7 and 8 are optimum values, those are too many clusters for the given dataset. So, I have chosen 6 as the number of clusters to group.



# final model with k=6
kmeans = KMeans(n_clusters=6, max_iter=50,random_state=0)
kmeans.fit(df1)

#random_state ensures that the labels do not change everytime the code is executed

Now we compare the overall dataset and label each candidate with the k-means group number





We can see that the column 'cluster_id' tells us which group the candidate belongs to. In this model there are 6 different group. Now, after manually observing the ideal group, this is how I grouped them:




 
 #Defining a fucntion
 def rec(x):
    if x==1:
        return "Highly Recommended Candidate"
    elif x==4:
        return "moderately Recommended Candidate"
    elif x==2:
        return "Low recommended candidate"
    else:
        return "Not recommended"
        
  #Applying the function on the dataset
  
  df1['recommend']=df1['cluster_id'].apply(rec)
  





The group with id_cluster =1 has the least distance, more experience, high resume score and high skill percentage as well. Then it is followed by group with cluster_id =4 and then cluster_id=2. Rest are not recommended at all.


And that is how the candidate system can help recruiters save time and effort of going through millions of resumes. To differentiate the candidate further, we can rank them on them on their resume score and arrange them according to their percentile score like this:



#Creating percentile scores based on resume score
df1['Total_Score']=df1["ResumeScore0"].rank(pct=True)*100

#Arranging the values by percentile score
df1=df1.sort_values(by=['Total_Score'],ascending=False)

#Viewing the candidates that are recommended for the job
df1[(df1['recommend']!='Not recommended')].head()






Critique of the model

The model is no way perfect and could use more modifications. Although, the model can be used to predict the right candidate based on any job description, it lacks a few features:


The model could use better features like parsing LinkedIn profiles, personal portfolio links and GitHub profiles which are sites that recruiters check for. Our website did not do that nor did I ever parse those links to look for important keywords.

In general, a mean silhouette score of around 0.5 or higher is considered to be good, but this will depend on the nature of the data and the desired level of separation between the clusters. All the silhouette score for this dataset are under 0.5. Also, the cluster with highest number of silhouette score was not chose for this model as they were too many clusters.

The model was made so that one does not have to come up with with their own metrics to identify the right candidate. Although, one could make a weighted linear model which can be used to predict the best candidate but the catch is that one has to set weights for their needs but if done right, the model can recommend the best candidate in simpler and more personalized way which doesn't require k-means.


Links and references 


The job portal was made using Bubble.io. The domain name no longer works with the bubble app.

     2. The python code was executed using https://anvil.works/  
