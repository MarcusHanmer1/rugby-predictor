# Cplt and web help used

import pandas as pd
import numpy as np

import streamlit as st

import joblib

from tensorflow.keras.models import load_model

import time

st.set_page_config(
    page_title="Rugby Predictor AI",
    layout="wide"
)

#load everything? ( no other way to say it!)
def load_everything():
    model = load_model('rugby_predictor.keras')
    scaler = joblib.load('scaler.joblib')
    data = pd.read_csv('final_data.csv', parse_dates=['date'])
    return model, scaler, data

model_lstm, scaler, data = load_everything()

all_teams_in_data = sorted(list(pd.concat([data['home_team'], data['away_team']]).unique()))

# wc pools
wc_pools = {
    "Pool A": ["New Zealand", "France", "Italy", "Uruguay", "Namibia"],
    "Pool B": ["South Africa", "Ireland", "Scotland", "Tonga", "Romania"],
    "Pool C": ["Wales", "Australia", "Fiji", "Georgia", "Portugal"],
    "Pool D": ["England", "Argentina", "Japan", "Samoa", "Chile"]
}

wc_teams = sorted(list(set([team for pool in wc_pools.values() for team in pool])))


qf_bracket = {
    "QF1": {"winner": "Pool C", "runner_up": "Pool D"},
    "QF2": {"winner": "Pool B", "runner_up": "Pool A"},
    "QF3": {"winner": "Pool D", "runner_up": "Pool C"},
    "QF4": {"winner": "Pool A", "runner_up": "Pool B"}
}

#fix for teams not in data!
tier_two = {
    'ranking': 65.0,
    'form': 0.5,
    'win_perc': 0.5,
    'rust': 90.0
}

#get stats from inputted teams
def get_team_stats(team, data_df):
   
    try:
        last_match = data_df[(data_df['home_team'] == team) | (data_df['away_team'] == team)].sort_values(by='date').iloc[-1]
        
        if last_match['home_team'] == team:
            stats = {
                'ranking': last_match['home_team_ranking_points'],
                'form': last_match['home_team_form_perc'],
                'win_perc': last_match['home_team_win_perc'],
                'rust': last_match['home_team_rust_days']
            }
        else:
            stats = {
                'ranking': last_match['away_team_ranking_points'],
                'form': last_match['away_team_form_perc'],
                'win_perc': last_match['away_team_win_perc'],
                'rust': last_match['away_team_rust_days']
            }
        return stats
    
    except IndexError:
        #catch tier two teams
        return tier_two

def get_h2h_stats(team_a, team_b, data_df):

    h2h_match = data_df[
        ((data_df['home_team'] == team_a) & (data_df['away_team'] == team_b)) |
        ((data_df['home_team'] == team_b) & (data_df['away_team'] == team_a))
    ].sort_values(by='date')
    
    if len(h2h_match) == 0:
        return 0.5, 0.5
    else:
        last_h2h_match = h2h_match.iloc[-1]
        
        team_a_h2h = last_h2h_match['home_team_h2h_win_perc'] if last_h2h_match['home_team'] == team_a else last_h2h_match['away_team_h2h_win_perc']
        team_b_h2h = last_h2h_match['home_team_h2h_win_perc'] if last_h2h_match['home_team'] == team_b else last_h2h_match['away_team_h2h_win_perc']
        
        return team_a_h2h, team_b_h2h

def predict_match(team_a, team_b, is_neutral, is_world_cup, model, scaler, data_df):

    stats_a = get_team_stats(team_a, data_df)
    stats_b = get_team_stats(team_b, data_df)
    
    h2h_a, h2h_b = get_h2h_stats(team_a, team_b, data_df)
    
    X_live = np.array([
        stats_a['ranking'] - stats_b['ranking'],
        stats_a['form'] - stats_b['form'],
        h2h_a - h2h_b,
        stats_a['rust'] - stats_b['rust'],
        stats_a['rust'],
        stats_b['rust'],
        stats_a['win_perc'],
        stats_b['win_perc'],
        int(is_neutral),
        int(is_world_cup)
    ])
    
    X_scaled = scaler.transform(X_live.reshape(1, -1))
    
    X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    
    #predict
    prediction = model.predict(X_lstm, verbose=0)[0][0]
    
    return prediction


#the app

st.title("Rugby predictor")
st.subheader("Using an LSTM neural network to predict international matches and simulate tournaments.")


tab1, tab2 = st.tabs(["**Single match predictor**", "**World cup simulator**"])

#single match
with tab1:
    st.header("Predict a single match")

    col1, col2, col3 = st.columns(3)
    with col1:
        team_a = st.selectbox("Select team A (home)", wc_teams, index=wc_teams.index("New Zealand"), key="single_team_a")
    with col2:
        team_b = st.selectbox("Select team B (away)", wc_teams, index=wc_teams.index("South Africa"), key="single_team_b")
    with col3:
        is_neutral = st.checkbox("Neither home or away?", value=False, key="single_neutral")
        is_world_cup = st.checkbox("World Cup match?", value=False, key="single_wc")

    if st.button("Predict single match", type="primary"):
        if team_a == team_b:
            st.error("Please select two different teams.")
        else:
            with st.spinner("Calculating..."):
                prediction_prob = predict_match(team_a, team_b, is_neutral, is_world_cup, model_lstm, scaler, data)
                
                prob_a = prediction_prob * 100
                prob_b = (1 - prediction_prob) * 100
                
                st.subheader("Prediction:")
                col1_res, col2_res = st.columns(2)
                with col1_res:
                    st.metric(label=f"**{team_a} (home) Win probability**", value=f"{prob_a:.1f}%")
                with col2_res:
                    st.metric(label=f"**{team_b} (away) Win probability**", value=f"{prob_b:.1f}%")

                if prob_a > prob_b:
                    st.success(f"**The model predicts {team_a} will win.**")
                else:
                    st.info(f"**The model predicts {team_b} will win.**")

#world cup
with tab2:
    st.header("Rugby World Cup simulator")
    st.markdown("This will simulate all 48 matches of the 2023 RWC using our trained model. We impute a default 'tier 2' profile for teams not in our training data.")
    
    if st.button("Simulate World Cup", type="primary"):
        pool_results = {}
        pool_tables = {}
        
        #pools
        with st.spinner("Simulating pool stage (40 Matches)..."):
            for pool_name, teams in wc_pools.items():
                st.subheader(f"Simulating {pool_name}...")
                results = []
                table = {team: {'Points': 0, 'W': 0, 'L': 0} for team in teams}
                
                for i in range(len(teams)):
                    for j in range(i + 1, len(teams)):
                        team_a = teams[i]
                        team_b = teams[j]
                        
                        prob_a_wins = predict_match(team_a, team_b, True, True, model_lstm, scaler, data)
                        
                        if prob_a_wins > 0.5:
                            winner, loser = team_a, team_b
                            table[team_a]['Points'] += 4
                            table[team_a]['W'] += 1
                            table[team_b]['L'] += 1
                        else:
                            winner, loser = team_b, team_a
                            table[team_b]['Points'] += 4
                            table[team_b]['W'] += 1
                            table[team_a]['L'] += 1
                        
                        results.append(f"{team_a} vs {team_b} -> **Winner: {winner}**")
                
                pool_results[pool_name] = results
                pool_tables[pool_name] = table
                time.sleep(0.5)

        st.success("Pool stage simulation complete!")

        st.header("Pool stage results")
        st.markdown("Teams are ranked by points")
        
        knockout_teams = {}
        cols = st.columns(4)
        
        for i, (pool_name, table) in enumerate(pool_tables.items()):
            with cols[i]:
                st.subheader(pool_name)
                pool_df = pd.DataFrame.from_dict(table, orient='index')
                pool_df = pool_df.sort_values(by='Points', ascending=False)
                st.dataframe(pool_df)
                
                knockout_teams[f"Winner {pool_name}"] = pool_df.index[0]
                knockout_teams[f"Runner-up {pool_name}"] = pool_df.index[1]

        #quarter finals
        st.header("Knockout stage simulation")
        ko_winners = {}
        
        with st.spinner("Simulating Quarter finals..."):
            qf_results = []
            
            team_a, team_b = knockout_teams["Winner Pool C"], knockout_teams["Runner-up Pool D"]
            prob_a = predict_match(team_a, team_b, True, True, model_lstm, scaler, data)
            ko_winners['QF1'] = team_a if prob_a > 0.5 else team_b
            qf_results.append(f"**QF1**: {team_a} vs {team_b} -> **Winner: {ko_winners['QF1']}**")

            team_a, team_b = knockout_teams["Winner Pool B"], knockout_teams["Runner-up Pool A"]
            prob_a = predict_match(team_a, team_b, True, True, model_lstm, scaler, data)
            ko_winners['QF2'] = team_a if prob_a > 0.5 else team_b
            qf_results.append(f"**QF2**: {team_a} vs {team_b} -> **Winner: {ko_winners['QF2']}**")
            
            team_a, team_b = knockout_teams["Winner Pool D"], knockout_teams["Runner-up Pool C"]
            prob_a = predict_match(team_a, team_b, True, True, model_lstm, scaler, data)
            ko_winners['QF3'] = team_a if prob_a > 0.5 else team_b
            qf_results.append(f"**QF3**: {team_a} vs {team_b} -> **Winner: {ko_winners['QF3']}**")

            team_a, team_b = knockout_teams["Winner Pool A"], knockout_teams["Runner-up Pool B"]
            prob_a = predict_match(team_a, team_b, True, True, model_lstm, scaler, data)
            ko_winners['QF4'] = team_a if prob_a > 0.5 else team_b
            qf_results.append(f"**QF4**: {team_a} vs {team_b} -> **Winner: {ko_winners['QF4']}**")

            st.subheader("Quarter final results")
            for res in qf_results:
                st.markdown(f"- {res}")
            time.sleep(0.5)

        #semi finals
        with st.spinner("Simulating Semi finals..."):
            sf_results = []
            
            team_a, team_b = ko_winners['QF1'], ko_winners['QF2']
            prob_a = predict_match(team_a, team_b, True, True, model_lstm, scaler, data)
            ko_winners['SF1'] = team_a if prob_a > 0.5 else team_b
            sf_results.append(f"**SF1**: {team_a} vs {team_b} -> **Winner: {ko_winners['SF1']}**")

            team_a, team_b = ko_winners['QF3'], ko_winners['QF4']
            prob_a = predict_match(team_a, team_b, True, True, model_lstm, scaler, data)
            ko_winners['SF2'] = team_a if prob_a > 0.5 else team_b
            sf_results.append(f"**SF2**: {team_a} vs {team_b} -> **Winner: {ko_winners['SF2']}**")
            
            st.subheader("Semi final results")
            for res in sf_results:
                st.markdown(f"- {res}")
            time.sleep(0.5)

        #final!
        with st.spinner("Simulating the FINAL..."):
            final_results = []
            
            team_a, team_b = ko_winners['SF1'], ko_winners['SF2']
            prob_a = predict_match(team_a, team_b, True, True, model_lstm, scaler, data)
            final_winner = team_a if prob_a > 0.5 else team_b
            final_results.append(f"**Final**: {team_a} vs {team_b} -> **Winner: {final_winner}**")

            st.subheader("Final Result")
            for res in final_results:
                st.markdown(f"- {res}")

        st.balloons()
        st.header(f"Predicted 2023 RWC champion: {final_winner}")