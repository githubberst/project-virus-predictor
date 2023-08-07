import streamlit as st
from streamlit_option_menu import option_menu

import pickle
import pandas as pd
import numpy as np

from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objs as go
import altair as alt
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

import re
import string
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# import dataset
df = pd.read_csv('data/final_df.csv',lineterminator='\n')

# load the models
model = pickle.load(open('models/model.pkl', 'rb'))
with open('models/pipe_gridsearch.pkl', 'rb') as pipe_gridsearch: # open a file, where you stored the pickled data
    pipe_gridsearch = pickle.load(pipe_gridsearch)

# definitions / lists
consumer_preference_options = ['Keto','Paleo']

stopwords = nltk.corpus.stopwords.words('english')
# note: we ran the EDA on top 20 grams once, and engineered our stopwords further
# this is the eventual list of stopwords that we will use for this second run
extrawords = ['pinned','top','subreddit','ask','question','ha','beginner','hello',
              'talk','wa','also','ive','im','use','community', 'would','think','really',\
              'nutshell', 'utshell', 'support','thread']
stopwords.extend(extrawords)
wn = WordNetLemmatizer()


# Define function to clean text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove r/
    text = text.replace('r/', '')
    # Remove &amp; 
    text = text.replace('&amp;', '')
    # Remove \n
    text = text.replace('\n', ' ')
    # Remove strings that contain http
    text = re.sub(r'\S*http\S*', '', text)
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Remove non-Roman characters
    text = re.sub('([^\x00-\x7F])+', ' ', text)
    # Remove chunks of empty spaces
    text = text.replace('\s+', ' ')
    # Remove 'nan'
    text = text.replace('nan', ' ')
    tokens = re.split('\W+', text)
    text = ' '.join([wn.lemmatize(word) for word in tokens if word not in stopwords])
    return text

def run_analysis(txt):
    df = pd.DataFrame()
    df['txt'] = [txt]
    result = pipe_gridsearch.predict(df['txt'])
    if result == ['keto']:
        result = 'Keto'
    else:
        result = 'Paleo'
    return result

def top_n_grams(corpus, n=20, ngram=(1,1), stop=None):
    # Create a CountVectorizer object with specified n-gram range and stop words
    vec = CountVectorizer(ngram_range=ngram, stop_words=stop)
    # Convert the corpus into a bag of words representation
    bag_of_words = vec.fit_transform(corpus)
    # Calculate the sum of words across all documents
    sum_words = bag_of_words.sum(axis=0) 
    # Create a list of (word, count) pairs from the vocabulary and word counts
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    # Sort the list of (word, count) pairs by count in descending order and select top n
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)[:n]
    # Store top n common n-grams in a dataframe
    df = pd.DataFrame(words_freq, columns=['text', 'count'])
    # Sort the dataframe by the count column in descending order
    df = df.sort_values(by='count', ascending=False)

    return df 

# streamlit shell (layouts etc)
# set webpage name and icon
st.set_page_config(
    page_title='Consumer Dietary Preferences: An Analysis',
    page_icon='🥘',
    layout='wide',
    initial_sidebar_state='expanded'
    )

# top navigation bar
selected = option_menu(
    menu_title = None,
    options = ['Insights', 'Businesses','Consumers'],
    icons = ['eyeglasses','bar-chart','chat-left-text'],
    default_index = 0, # which tab it should open when page is first loaded
    orientation = 'horizontal',
    styles={
        'nav-link-selected': {'background-color': '#FF7F0E'}
        }
    )

if selected == 'Insights':
    # title
    st.title('Consumer Dietary Preferences: An Analysis')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    # keywords
    st.text('') # add space
    st.header('Keywords')

    # slider
    st.subheader('Top n Frequently Occurring Words')
    top_n_words = st.slider('Select n', 6, 20, 12)
    # create df for top n words
    keto_repeat_uni_df = top_n_grams(df[df['redditlabel'] == 0]['text_lemma'], n=top_n_words, ngram=(1, 1), stop=stopwords)
    paleo_repeat_uni_df = top_n_grams(df[df['redditlabel'] == 1]['text_lemma'], n=top_n_words, ngram=(1, 1), stop=stopwords)
    keto_repeat_bi_df = top_n_grams(df[df['redditlabel'] == 0]['text_lemma'], n=top_n_words, ngram=(2, 2), stop=stopwords)
    paleo_repeat_bi_df = top_n_grams(df[df['redditlabel'] == 1]['text_lemma'], n=top_n_words, ngram=(2, 2), stop=stopwords)
    # set 'text as index'
    keto_repeat_uni_series = keto_repeat_uni_df.set_index('text')
    keto_repeat_bi_series = keto_repeat_bi_df.set_index('text')
    paleo_repeat_uni_series = paleo_repeat_uni_df.set_index('text')
    paleo_repeat_bi_series = paleo_repeat_bi_df.set_index('text')
    # convert series to dict
    keto_uni_dict = keto_repeat_uni_series[0:top_n_words].to_dict()
    keto_bi_dict = keto_repeat_bi_series[0:top_n_words].to_dict()
    paleo_uni_dict = paleo_repeat_uni_series[0:top_n_words].to_dict()
    paleo_bi_dict = paleo_repeat_bi_series[0:top_n_words].to_dict()
    # dict in dict, so extract the inner dict in key 'count'
    keto_uni_dict = keto_uni_dict['count']
    keto_bi_dict = keto_bi_dict['count']
    paleo_uni_dict = paleo_uni_dict['count']
    paleo_bi_dict = paleo_bi_dict['count']
    
    # create keto uni plot
    fig_keto_uni, axes_keto_uni = plt.subplots(1,1,facecolor='None')
    
    # keto uni word cloud
    wc_keto_uni = WordCloud(
        mode='RGBA',
        background_color = None,
        height = 2000,
        width = 3000,
        colormap='Blues'
        )
    wc_keto_uni = wc_keto_uni.generate_from_frequencies(keto_uni_dict)
    axes_keto_uni.imshow(wc_keto_uni)
    axes_keto_uni.axis('off')

    # create keto bi plot
    fig_keto_bi, axes_keto_bi = plt.subplots(1,1,facecolor='None')
    
    # keto bigram word cloud
    wc_keto_bi = WordCloud(
        mode='RGBA',
        background_color = None,
        height = 2000,
        width = 3000,
        colormap='Blues'
        )
    
    wc_keto_bi = wc_keto_bi.generate_from_frequencies(keto_bi_dict)
    axes_keto_bi.imshow(wc_keto_bi)
    axes_keto_bi.axis('off')

    # create paleo uni plot
    fig_paleo_uni, axes_paleo_uni = plt.subplots(1,1,facecolor='None')   

    # paleo word cloud
    wc_paleo_uni = WordCloud(
        mode='RGBA',
        background_color = None,
        height = 2000,
        width = 3000,
        colormap='Wistia'
        )
    wc_paleo_uni = wc_paleo_uni.generate_from_frequencies(paleo_uni_dict)
    axes_paleo_uni.imshow(wc_paleo_uni)
    axes_paleo_uni.axis('off')

    # create paleo bi plot
    fig_paleo_bi, axes_paleo_bi = plt.subplots(1,1,facecolor='None')   

    # paleo word cloud
    wc_paleo_bi = WordCloud(
        mode='RGBA',
        background_color = None,
        height = 2000,
        width = 3000,
        colormap='Wistia'
        )
    wc_paleo_bi = wc_paleo_bi.generate_from_frequencies(paleo_bi_dict)
    axes_paleo_bi.imshow(wc_paleo_bi)
    axes_paleo_bi.axis('off')


    # keto uni barchart
    keto_uni_bar_chart = px.bar(
        keto_repeat_uni_df[0:top_n_words].sort_values(by='count'),
        x='count',
        y='text',
        orientation='h',
        title=f'<b>Top {top_n_words} Frequently Occurring Words in r/Keto</b>',
        color_discrete_sequence = ['#1F77B4'] * len(keto_repeat_uni_df[0:top_n_words]),
        )
    keto_uni_bar_chart.update_layout(
        autosize=False,
        height=500)

    # keto bi barchart
    keto_bi_bar_chart = px.bar(
        keto_repeat_bi_df[0:top_n_words].sort_values(by='count'),
        x='count',
        y='text',
        orientation='h',
        title=f'<b>Top {top_n_words} Frequently Occurring Words in r/Keto</b>',
        color_discrete_sequence = ['#1F77B4'] * len(keto_repeat_bi_df[0:top_n_words]),
        )
    keto_bi_bar_chart.update_layout(
        autosize=False,
        height=500)

    # paleo uni barchart
    paleo_uni_bar_chart = px.bar(
        paleo_repeat_uni_df[0:top_n_words].sort_values(by='count'),
        x='count',
        y='text',
        orientation='h',
        title=f'<b>Top {top_n_words} Frequently Occurring Words in r/Paleo</b>',
        color_discrete_sequence = ['#FF7F0E'] * len(paleo_repeat_uni_df[0:top_n_words])
        )
    paleo_uni_bar_chart.update_layout(
        autosize=False,
        height=500)

    # paleo bi barchart
    paleo_bi_bar_chart = px.bar(
        paleo_repeat_bi_df[0:top_n_words].sort_values(by='count'),
        x='count',
        y='text',
        orientation='h',
        title=f'<b>Top {top_n_words} Frequently Occurring Words in r/Paleo</b>',
        color_discrete_sequence = ['#FF7F0E'] * len(paleo_repeat_bi_df[0:top_n_words])
        )
    paleo_bi_bar_chart.update_layout(
        autosize=False,
        height=500)
    
    # word cloud
    gram_tab1, gram_tab2 = st.tabs(['1 Word', '2 Word'])
    with gram_tab1:
        # create columns to separate keto and paleo
        left_column, right_column  = st.columns(2)

        # left col - Keto
        left_column.subheader('Keto')
        # right cols - Paleo
        right_column.subheader('Paleo')
        
        # word cloud - keto
        left_column.pyplot(fig_keto_uni)
        # word cloud - paleo
        right_column.pyplot(fig_paleo_uni)

        # bar chart - keto
        left_column.plotly_chart(keto_uni_bar_chart,use_container_width=True)
        # bar chart - paleo
        right_column.plotly_chart(paleo_uni_bar_chart,use_container_width=True)

        # text explaination for frequently appearing words
        st.write('Distinct Interests:')
        st.write('The r/Keto community focuses on macronutrient composition, particularly \
    carbohydrates, fats, and proteins, to achieve and maintain ketosis. \
    In contrast, r/Paleo emphasizes specific ingredients and unprocessed foods that mimic \
    ancestral diets.')

        st.text("") # add extra line in between

        # common words
        st.header('Common Words found in both r/Keto and r/Paleo')

        # venn diagram
        fig_venn_uni, ax_venn_uni = plt.subplots(facecolor='None')
        
        keto_venn_uni = set(keto_repeat_uni_df['text']) # keto
        paleo_venn_uni = set(paleo_repeat_uni_df['text']) #paleo

        colors = ['#1F77B4', '#FF7F0E', 'orange']

        venn_uni = venn2([keto_venn_uni, paleo_venn_uni],set_labels=['Keto','Paleo'],ax=ax_venn_uni, set_colors=colors, alpha=0.9)
        venn_uni.get_label_by_id('100').set_text('\n'.join(map(str,keto_venn_uni - paleo_venn_uni)),)
        venn_uni.get_label_by_id('110').set_text('\n'.join(map(str,keto_venn_uni & paleo_venn_uni)))
        venn_uni.get_label_by_id('010').set_text('\n'.join(map(str,paleo_venn_uni - keto_venn_uni)))

        venn_uni.get_label_by_id('A').set_size(15)
        venn_uni.get_label_by_id('B').set_size(15)
        venn_uni.get_label_by_id('100').set_size(8)
        venn_uni.get_label_by_id('110').set_size(8)
        venn_uni.get_label_by_id('010').set_size(8)

        
        # change set label color to match color of circle
        i = 0
        for text in venn_uni.set_labels:
            text.set_color(colors[i])
            i+=1

        for text in venn_uni.subset_labels:
            text.set_color('black')
        st.pyplot(fig_venn_uni)

    # text explanation for common words
    st.write('Common Interests:')
    st.write('Upon examining the shared words and bigrams within the r/Keto and r/Paleo \
subreddits, it is evident that both communities demonstrate a profound interest in \
their daily dietary practices and actively seek additional perspectives through the \
forum. Notably, weight loss and low carbohydrate intake emerge as significant areas \
of emphasis for both subreddits.')
    
    with gram_tab2:
        # create columns to separate keto and paleo
        left_column, right_column  = st.columns(2)
        
        # left col - Keto
        left_column.subheader('Keto')
        # right cols - Paleo
        right_column.subheader('Paleo')
        
        # word cloud - keto
        left_column.pyplot(fig_keto_bi)
        # word cloud - paleo
        right_column.pyplot(fig_paleo_bi)
        
        # bar chart - keto
        left_column.plotly_chart(keto_bi_bar_chart,use_container_width=True)
        # bar chart - paleo
        right_column.plotly_chart(paleo_bi_bar_chart,use_container_width=True)    

        # venn diagram
        fig_venn_bi, ax_venn_bi = plt.subplots(facecolor='None')
        
        keto_venn_bi = set(keto_repeat_bi_df['text']) # keto
        paleo_venn_bi = set(paleo_repeat_bi_df['text']) #paleo

        colors = ['#1F77B4', '#FF7F0E', 'orange']

        venn_bi = venn2([keto_venn_bi, paleo_venn_bi],set_labels=['Keto','Paleo'],ax=ax_venn_bi, set_colors=colors, alpha=0.9)
        venn_bi.get_label_by_id('100').set_text('\n'.join(map(str,keto_venn_bi - paleo_venn_bi)),)
        venn_bi.get_label_by_id('110').set_text('\n'.join(map(str,keto_venn_bi & paleo_venn_bi)))
        venn_bi.get_label_by_id('010').set_text('\n'.join(map(str,paleo_venn_bi - keto_venn_bi)))

        venn_bi.get_label_by_id('A').set_size(15)
        venn_bi.get_label_by_id('B').set_size(15)
        venn_bi.get_label_by_id('100').set_size(8)
        venn_bi.get_label_by_id('110').set_size(8)
        venn_bi.get_label_by_id('010').set_size(8)
        
        # change set label color to match color of circle
        i = 0
        for text in venn_bi.set_labels:
            text.set_color(colors[i])
            i+=1

        for text in venn_bi.subset_labels:
            text.set_color('black')
        st.pyplot(fig_venn_bi)



if selected == 'Businesses':
    # title
    st.title('Consumer Dietary Preferences: An Analysis')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    st.header('Analyze your customers!')
    data = st.file_uploader('Upload CSV file')
    if data is not None:
        df = pd.read_csv(data)
        st.success('Submitted')

        df['clean'] = df['titletext'].map(clean_text)
        y_pred = model.predict(df['clean'].values)
        unique, counts = np.unique(y_pred, return_counts=True)

        col1, col2 = st.columns([1, 3])
        with col2:
            colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
            
            fig = go.Figure(go.Pie(
                labels=['Keto', 'Paleo'],
                values=counts,
                marker=dict(colors=colors),
                textinfo='percent+label',  # Show percentage and label inside the pie slices
                showlegend=True))  # Adjust the legend font size

            fig.update_traces(textfont=dict(size=20))  # Adjust the font size of the percentage labels inside the pie slices
            fig.update_layout(
                title={'text':"Customer's Diet Distribution",
                        'font': {'size': 24}},
                height=800,  # Adjust the height of the figure
                width=800, # Adjust the width of the figure
                legend=dict(font=dict(size=20))) 
            st.plotly_chart(fig)  

        # Setting 3 tabs for unigram, bigram, trigram for the horizontal barcharts
        gram_tab1, gram_tab2 = st.tabs(['1 Word', '2 Word'])
        
        # Preloading the keto/paleo 
        df['predicted'] = model.predict(df['clean'].values)
        keto_df = df[df['predicted']==0]
        paleo_df = df[df['predicted']==1]

        words = st.slider('Number of words', min_value=5, max_value=20, step=1)

        with gram_tab1:
            col1, col2 = st.columns(2)
            with col1:
                cvec = CountVectorizer(max_df=0.85, min_df=2, max_features=5000, ngram_range=(1,1))
                X_keto = cvec.fit_transform(keto_df['clean'])
                X_keto_df = pd.DataFrame(X_keto.toarray(), columns=cvec.get_feature_names_out())
                bigram_keto = X_keto_df.sum().sort_values(ascending=False).head(words)[::-1]
                fig = px.bar(bigram_keto,
                            x=bigram_keto,
                            y=bigram_keto.index,
                            orientation='h')

                fig.update_traces(marker_color='#1f77b4') # Set the color of "Keto" bar to blue

                fig.update_layout(
                    title={'text':'Top Unigram Keywords (Keto)',
                            'font': {'size': 24}},
                    xaxis={'title':'Count', # Setting title of x axis label
                            'title_font':dict(size=18), # Changing font size x axis label title
                            'tickfont':dict(size=16) # Changing font size of axis and tick labels
                            },
                    yaxis={'title':'Keywords', # Setting title of y axis label
                            'title_font':dict(size=18), # Changing font size y axis label title
                            'tickfont':dict(size=16) # Changing font size of axis and tick labels
                            },
                    height=600)

                st.plotly_chart(fig)

            with col2:
                cvec = CountVectorizer(max_df=0.85, min_df=2, max_features=5000, ngram_range=(1,1))
                X_paleo = cvec.fit_transform(paleo_df['clean'])
                X_paleo_df = pd.DataFrame(X_paleo.toarray(), columns=cvec.get_feature_names_out())
                bigram_paleo = X_paleo_df.sum().sort_values(ascending=False).head(words)[::-1]
                fig = px.bar(bigram_paleo, x=bigram_paleo, y=bigram_paleo.index, orientation='h')

                fig.update_traces(marker_color='#ff7f0e') # Set the color of "Paleo" bar to orange
                fig.update_layout(
                    title={'text':'Top Unigram Keywords (Paleo)',
                            'font': {'size': 24}},
                    xaxis={'title':'Count',
                            'title_font':dict(size=18),
                            'tickfont':dict(size=16)
                            },
                    yaxis={'title':'Keywords',
                            'title_font':dict(size=18), 
                            'tickfont':dict(size=16) 
                            },
                    height=600)
    

                st.plotly_chart(fig)
        
        with gram_tab2:
            col1, col2 = st.columns(2)

            with col1:
                cvec = CountVectorizer(max_df=0.85, min_df=2, max_features=5000, ngram_range=(2,2))
                X_keto = cvec.fit_transform(keto_df['clean'])
                X_keto_df = pd.DataFrame(X_keto.toarray(), columns=cvec.get_feature_names_out())
                bigram_keto = X_keto_df.sum().sort_values(ascending=False).head(words)[::-1]
                fig = px.bar(bigram_keto, x=bigram_keto, y=bigram_keto.index, orientation='h')

                fig.update_traces(marker_color='#1f77b4') # Set the color of "Keto" bar to blue
                fig.update_layout(
                    title={'text':'Top Bigram Keywords (Keto)',
                            'font': {'size': 24}},
                    xaxis={'title':'Count',
                            'title_font':dict(size=18),
                            'tickfont':dict(size=16)
                            },
                    yaxis={'title':'Keywords',
                            'title_font':dict(size=18),
                            'tickfont':dict(size=16)
                            },
                    height=600)

                st.plotly_chart(fig)

            with col2:
                cvec = CountVectorizer(max_df=0.85, min_df=2, max_features=5000, ngram_range=(2,2))
                X_paleo = cvec.fit_transform(paleo_df['clean'])
                X_paleo_df = pd.DataFrame(X_paleo.toarray(), columns=cvec.get_feature_names_out())
                bigram_paleo = X_paleo_df.sum().sort_values(ascending=False).head(words)[::-1]
                fig = px.bar(bigram_paleo, x=bigram_paleo, y=bigram_paleo.index, orientation='h')

                fig.update_traces(marker_color='#ff7f0e') # Set the color of "Paleo" bar to orange
                fig.update_layout(
                    title={'text':'Top Bigram Keywords (Paleo)',
                            'font': {'size': 24}},
                    xaxis={'title':'Count',
                            'title_font':dict(size=18),
                            'tickfont':dict(size=14)
                            },
                    yaxis={'title':'Keywords',
                            'title_font':dict(size=18),
                            'tickfont':dict(size=14)
                            },
                    height=600)

                st.plotly_chart(fig)


if selected == 'Consumers':
    # title
    st.title('Consumer Dietary Preferences: An Analysis')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)
    
    st.header('Consumer View')
    st.subheader("Ask us anything!")
    user_comment = st.text_area('Enter here')
    
    # Cleaning the string
    cleaned_user_comment = clean_text(user_comment)
    prediction = model.predict(pd.Series(cleaned_user_comment))
    if st.button('Submit'):
        st.success('Successfully submitted!')
        if prediction == 1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image('images/paleo_puffs.png', width=400)
                st.markdown('<p style="font-size:20px; text-align:left;">Paleo Puffs</p>', unsafe_allow_html=True)
            with col2:
                st.image('images/paleo_chips.webp', width=300)
                st.markdown('<p style="font-size:20px; text-align:left;">Paleo Chips</p>', unsafe_allow_html=True)
            with col3:
                st.image('images/paleo_muesli.png', width=250)
                st.markdown('<p style="font-size:20px; text-align:left;">Paleo Muesli</p>', unsafe_allow_html=True)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image('images/keto_nut_mix.webp', width=400)
                st.markdown('<p style="font-size:20px; text-align:left;">Keto Cookies</p>', unsafe_allow_html=True)
            with col2:
                st.image('images/keto_cookies.webp', width=400)
                st.markdown('<p style="font-size:20px; text-align:left;">Keto Cookies</p>', unsafe_allow_html=True)
            with col3:
                st.image('images/keto_bar.webp', width=400)
                st.markdown('<p style="font-size:20px; text-align:left;">Keto Bar</p>', unsafe_allow_html=True)
