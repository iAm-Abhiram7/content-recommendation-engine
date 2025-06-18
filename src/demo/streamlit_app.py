"""
Streamlit Demo Application
Impressive, interactive demo interface for the recommendation system
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics.accuracy_metrics import MetricsEvaluator
from src.evaluation.testing.load_testing import quick_load_test
from src.utils.demo_data import DemoDataGenerator

# Configure page
st.set_page_config(
    page_title="Content Recommendation Engine - Production Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Status indicators styling with better contrast */
    .status-success {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white !important;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin: 0.2rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: none;
        font-size: 0.9rem;
    }
    
    .status-error {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white !important;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin: 0.2rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: none;
        font-size: 0.9rem;
    }
    
    .status-info {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white !important;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin: 0.2rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: none;
        font-size: 0.9rem;
    }
    
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .explanation-box {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        color: black !important;
    }
    
    .stTabs [data-baseweb="tab"] p {
        color: black !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] p {
        color: white !important;
    }
    
    /* Ensure tab text containers have proper colors */
    .stTabs [data-testid="stMarkdownContainer"] {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

class DemoApp:
    """Main demo application class"""
    
    def __init__(self):
        self.api_base_url = 'http://localhost:8000'  # Fixed API URL
        self.demo_data = DemoDataGenerator()
        self.metrics_evaluator = MetricsEvaluator()
        
        # Initialize session state
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        if 'recommendation_history' not in st.session_state:
            st.session_state.recommendation_history = []
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🎯 Content Recommendation Engine</h1>', unsafe_allow_html=True)
        st.markdown("### Production-Ready AI-Powered Recommendation System with Adaptive Learning")
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = self.check_api_health()
            if status['healthy']:
                st.markdown(f'<div class="status-success">🟢 API Status: {status["status"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-error">🔴 API Status: {status["status"]}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="status-info">🔄 Adaptive Learning: Active</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="status-info">🎯 Models: Hybrid (3 algorithms)</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="status-info">📊 Metrics: Real-time</div>', unsafe_allow_html=True)
    
    def check_api_health(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {'healthy': True, 'status': data.get('status', 'unknown')}
        except:
            pass
        return {'healthy': False, 'status': 'unavailable'}
    
    def render_sidebar(self):
        """Render sidebar with user controls"""
        st.sidebar.markdown("## 🎛️ User Profile & Preferences")
        
        # Add refresh button
        if st.sidebar.button("🔄 Refresh Status", key="refresh_status"):
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # User selection
        users = self.demo_data.get_sample_users()
        selected_user = st.sidebar.selectbox(
            "Select Demo User",
            options=list(users.keys()),
            format_func=lambda x: f"{users[x]['name']} ({users[x]['persona']})",
            key="user_selector"
        )
        
        if selected_user:
            user_data = users[selected_user]
            st.sidebar.json(user_data, expanded=False)
        else:
            user_data = None
        
        st.sidebar.markdown("---")
        
        # Preference controls
        st.sidebar.markdown("### 🎚️ Content Preferences")
        
        # Genre preferences
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Documentary', 'Animation']
        genre_preferences = {}
        
        for genre in genres:
            default_value = 0.5
            if user_data and 'preferences' in user_data:
                default_value = user_data['preferences'].get(genre.lower(), 0.5)
            
            genre_preferences[genre] = st.sidebar.slider(
                f"{genre}",
                min_value=0.0,
                max_value=1.0,
                value=default_value,
                step=0.1,
                key=f"genre_{genre}"
            )
        
        # Content type preferences
        st.sidebar.markdown("### 📺 Content Types")
        content_types = {
            'Movies': st.sidebar.slider("Movies", 0.0, 1.0, 0.7, 0.1, key="content_movies"),
            'TV Shows': st.sidebar.slider("TV Shows", 0.0, 1.0, 0.5, 0.1, key="content_tv"),
            'Books': st.sidebar.slider("Books", 0.0, 1.0, 0.3, 0.1, key="content_books"),
            'Music': st.sidebar.slider("Music", 0.0, 1.0, 0.6, 0.1, key="content_music"),
            'Podcasts': st.sidebar.slider("Podcasts", 0.0, 1.0, 0.4, 0.1, key="content_podcasts")
        }
        
        # Exploration vs Exploitation
        st.sidebar.markdown("### ⚖️ Discovery Settings")
        exploration_factor = st.sidebar.slider(
            "Exploration vs Familiarity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            help="0 = Safe recommendations, 1 = Adventurous discoveries",
            key="exploration_slider"
        )
        
        diversity_preference = st.sidebar.slider(
            "Diversity Preference",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="How diverse should recommendations be?",
            key="diversity_slider"
        )
        
        # Store preferences in session state
        st.session_state.user_preferences = {
            'user_id': selected_user,
            'user_data': user_data,
            'genre_preferences': genre_preferences,
            'content_types': content_types,
            'exploration_factor': exploration_factor,
            'diversity_preference': diversity_preference
        }
        
        return selected_user, user_data
    
    def render_main_dashboard(self, user_id: str, user_data: Dict[str, Any]):
        """Render main dashboard with recommendations"""
        # Main content r
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Recommendations", 
            "📊 Performance Metrics", 
            "🔍 Explainable AI", 
            "⚡ Real-time Adaptation",
            "🧪 A/B Testing",
            "⚖️ Fairness Monitor"
        ])
        
        with tab1:
            self.render_recommendations_tab(user_id, user_data)
        
        with tab2:
            self.render_metrics_tab()
        
        with tab3:
            self.render_explanation_tab()
        
        with tab4:
            self.render_adaptation_tab()
        
        with tab5:
            self.render_testing_tab()
        
        with tab6:
            self.render_fairness_tab()
    
    def render_recommendations_tab(self, user_id: str, user_data: Dict[str, Any]):
        """Render recommendations tab"""
        st.markdown("## 🎯 Multi-Domain Recommendations")
        
        # Generate recommendations button
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🎯 Generate Recommendations", type="primary", use_container_width=True):
                with st.spinner("Generating personalized recommendations..."):
                    recommendations = self.generate_recommendations(user_id)
                    st.session_state['latest_recommendations'] = recommendations
        
        with col2:
            refresh_auto = st.checkbox("Auto-refresh", help="Automatically refresh recommendations")
        
        with col3:
            if st.button("🔄 Clear History"):
                st.session_state.recommendation_history = []
                st.success("History cleared!")
        
        # Display recommendations if available
        if 'latest_recommendations' in st.session_state:
            recommendations = st.session_state['latest_recommendations']
            
            # Create tabs for different content types
            movie_tab, book_tab, music_tab, cross_domain_tab = st.tabs([
                "🎬 Movies", "📚 Books", "🎵 Music", "🔀 Cross-Domain"
            ])
            
            with movie_tab:
                self.render_content_recommendations(recommendations.get('movies', []), 'movie')
            
            with book_tab:
                self.render_content_recommendations(recommendations.get('books', []), 'book')
            
            with music_tab:
                self.render_content_recommendations(recommendations.get('music', []), 'music')
            
            with cross_domain_tab:
                self.render_cross_domain_recommendations(recommendations)
    
    def render_content_recommendations(self, items: List[Dict], content_type: str):
        """Render recommendations for a specific content type"""
        if not items:
            st.info(f"No {content_type} recommendations available. Click 'Generate Recommendations' to get started.")
            return
        
        for i, item in enumerate(items[:6]):  # Show top 6
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    # Placeholder for content image
                    st.image(f"https://via.placeholder.com/150x200/1f77b4/ffffff?text={content_type.upper()}", 
                            width=100)
                
                with col2:
                    st.markdown(f"### {item.get('title', 'Unknown Title')}")
                    st.markdown(f"**Genre:** {item.get('genre', 'Unknown')}")
                    st.markdown(f"**Year:** {item.get('year', 'Unknown')}")
                    
                    # Recommendation score
                    score = item.get('score', 0.5)
                    st.progress(score, text=f"Recommendation Score: {score:.2f}")
                    
                    # Component breakdown
                    components = item.get('components', {})
                    if components:
                        st.markdown("**Score Breakdown:**")
                        for component, value in components.items():
                            st.markdown(f"- {component.title()}: {value:.2f}")
                
                with col3:
                    # Feedback buttons
                    col_like, col_dislike = st.columns(2)
                    
                    with col_like:
                        if st.button("👍", key=f"like_{content_type}_{i}"):
                            self.record_feedback(item['id'], 1, "like")
                            st.success("👍 Liked!")
                    
                    with col_dislike:
                        if st.button("👎", key=f"dislike_{content_type}_{i}"):
                            self.record_feedback(item['id'], -1, "dislike")
                            st.error("👎 Disliked!")
                    
                    # Not interested button
                    if st.button("❌ Not Interested", key=f"not_interested_{content_type}_{i}"):
                        self.record_feedback(item['id'], 0, "not_interested")
                        st.info("❌ Noted!")
                
                st.markdown("---")
    
    def render_cross_domain_recommendations(self, recommendations: Dict[str, List]):
        """Render cross-domain recommendations"""
        st.markdown("### 🔀 Cross-Domain Discoveries")
        st.markdown("*If you enjoyed this content, you might also like...*")
        
        # Create cross-domain connections
        movie_items = recommendations.get('movies', [])[:2]
        book_items = recommendations.get('books', [])[:2]
        music_items = recommendations.get('music', [])[:2]
        
        for i, (movie, book, music) in enumerate(zip(movie_items, book_items, music_items)):
            st.markdown(f"#### Discovery Set {i+1}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**🎬 Movie:** {movie.get('title', 'N/A')}")
                st.markdown(f"*{movie.get('genre', 'Unknown')} • {movie.get('year', 'Unknown')}*")
                
            with col2:
                st.markdown(f"**📚 Book:** {book.get('title', 'N/A')}")
                st.markdown(f"*{book.get('genre', 'Unknown')} • {book.get('year', 'Unknown')}*")
                
            with col3:
                st.markdown(f"**🎵 Music:** {music.get('title', 'N/A')}")
                st.markdown(f"*{music.get('genre', 'Unknown')} • {music.get('year', 'Unknown')}*")
            
            # Explanation for cross-domain connection
            st.markdown(f"""
            <div class="explanation-box">
                <strong>🧠 AI Explanation:</strong> These items share thematic elements of 
                {movie.get('genre', 'adventure')} and emotional resonance. The movie's 
                {movie.get('mood', 'uplifting')} tone connects to the book's narrative style 
                and the music's {music.get('mood', 'energetic')} energy.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    def render_metrics_tab(self):
        """Render performance metrics tab"""
        st.markdown("## 📊 Real-Time Performance Metrics")
        
        # Generate sample metrics
        metrics_data = self.generate_sample_metrics()
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ndcg_score = metrics_data['ndcg_at_10']
            color = "success" if ndcg_score > 0.35 else "warning"
            st.metric(
                "NDCG@10",
                f"{ndcg_score:.3f}",
                delta=f"{ndcg_score - 0.35:.3f}",
                delta_color="normal" if ndcg_score > 0.35 else "inverse"
            )
        
        with col2:
            diversity_score = metrics_data['diversity']
            color = "success" if diversity_score > 0.7 else "warning"
            st.metric(
                "Diversity",
                f"{diversity_score:.3f}",
                delta=f"{diversity_score - 0.7:.3f}",
                delta_color="normal" if diversity_score > 0.7 else "inverse"
            )
        
        with col3:
            latency = metrics_data['avg_latency_ms']
            color = "success" if latency < 100 else "warning"
            st.metric(
                "Avg Latency",
                f"{latency:.1f}ms",
                delta=f"{100 - latency:.1f}ms",
                delta_color="inverse" if latency < 100 else "normal"
            )
        
        with col4:
            throughput = metrics_data['throughput_rps']
            st.metric(
                "Throughput",
                f"{throughput:.0f} RPS",
                delta=f"{throughput - 500:.0f}",
                delta_color="normal"
            )
        
        # Detailed metrics charts
        col1, col2 = st.columns(2)
        
        with col1:
            # NDCG and Diversity over time
            time_data = self.generate_time_series_data()
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('NDCG@10 Over Time', 'Diversity Over Time'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_data['timestamp'],
                    y=time_data['ndcg'],
                    mode='lines+markers',
                    name='NDCG@10',
                    line=dict(color='#1f77b4')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_data['timestamp'],
                    y=time_data['diversity'],
                    mode='lines+markers',
                    name='Diversity',
                    line=dict(color='#ff7f0e')
                ),
                row=2, col=1
            )
            
            # Add target lines
            fig.add_hline(y=0.35, row=1, col=1, line_dash="dash", 
                         line_color="red", annotation_text="Target: 0.35")
            fig.add_hline(y=0.7, row=2, col=1, line_dash="dash", 
                         line_color="red", annotation_text="Target: 0.7")
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance distribution
            latency_data = np.random.lognormal(4, 0.5, 1000)  # Sample latency data
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=latency_data,
                nbinsx=30,
                name='Response Time Distribution',
                marker_color='#1f77b4'
            ))
            
            fig.add_vline(x=100, line_dash="dash", line_color="red", 
                         annotation_text="Target: 100ms")
            fig.add_vline(x=np.percentile(latency_data, 95), line_dash="dot", 
                         line_color="orange", annotation_text="P95")
            
            fig.update_layout(
                title="Response Time Distribution",
                xaxis_title="Response Time (ms)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # System health metrics
        st.markdown("### 🖥️ System Health")
        
        system_metrics = {
            'CPU Usage': 45.2,
            'Memory Usage': 67.8,
            'Disk Usage': 34.1,
            'Network I/O': 23.4
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        for i, (metric, value) in enumerate(system_metrics.items()):
            with [col1, col2, col3, col4][i]:
                color = "🟢" if value < 70 else "🟡" if value < 90 else "🔴"
                st.metric(f"{color} {metric}", f"{value:.1f}%")
    
    def render_explanation_tab(self):
        """Render explainable AI tab"""
        st.markdown("## 🔍 Explainable AI & Recommendation Insights")
        
        # Select a recommendation to explain
        if 'latest_recommendations' in st.session_state:
            recommendations = st.session_state['latest_recommendations']
            all_items = []
            
            for content_type, items in recommendations.items():
                for item in items[:3]:  # Top 3 from each category
                    all_items.append({
                        'display': f"{item.get('title', 'Unknown')} ({content_type})",
                        'item': item,
                        'type': content_type
                    })
            
            if all_items:
                selected_item = st.selectbox(
                    "Select recommendation to explain:",
                    options=all_items,
                    format_func=lambda x: x['display']
                )
                
                if selected_item:
                    self.render_recommendation_explanation(selected_item['item'], selected_item['type'])
        else:
            st.info("Generate recommendations first to see explanations.")
    
    def render_recommendation_explanation(self, item: Dict[str, Any], content_type: str):
        """Render detailed explanation for a specific recommendation"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(f"https://via.placeholder.com/200x300/1f77b4/ffffff?text={content_type.upper()}", 
                    width=150)
            
            st.markdown(f"### {item.get('title', 'Unknown Title')}")
            st.markdown(f"**Type:** {content_type.title()}")
            st.markdown(f"**Genre:** {item.get('genre', 'Unknown')}")
            st.markdown(f"**Year:** {item.get('year', 'Unknown')}")
            st.markdown(f"**Overall Score:** {item.get('score', 0.5):.3f}")
        
        with col2:
            st.markdown("### 🧠 AI Explanation")
            
            # Generate detailed explanation
            explanation = self.generate_explanation(item, content_type)
            st.markdown(explanation)
            
            # Component breakdown chart
            components = item.get('components', {
                'collaborative': 0.3,
                'content_based': 0.4,
                'knowledge_based': 0.2,
                'popularity': 0.1
            })
            
            fig = go.Figure(data=go.Bar(
                x=list(components.keys()),
                y=list(components.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ))
            
            fig.update_layout(
                title="Recommendation Component Breakdown",
                xaxis_title="Component",
                yaxis_title="Contribution Score",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Similar users and items
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👥 Similar Users")
            similar_users = self.generate_similar_users()
            for user in similar_users:
                st.markdown(f"- **{user['name']}** (similarity: {user['similarity']:.2f})")
                st.markdown(f"  *{user['common_interests']}*")
        
        with col2:
            st.markdown("### 📚 Similar Items")
            similar_items = self.generate_similar_items(content_type)
            for similar_item in similar_items:
                st.markdown(f"- **{similar_item['title']}** (similarity: {similar_item['similarity']:.2f})")
                st.markdown(f"  *{similar_item['reason']}*")
        
        # Reasoning flow
        st.markdown("### 🔄 Reasoning Flow")
        
        with st.expander("View detailed reasoning process"):
            reasoning_steps = [
                "🔍 **User Profile Analysis**: Analyzed your viewing history of 127 items",
                "👥 **Collaborative Filtering**: Found 15 users with similar preferences",
                "📊 **Content Analysis**: Matched genre preferences (Action: 0.8, Sci-Fi: 0.6)",
                "🧠 **Knowledge Graph**: Connected through 'space thriller' concept",
                "⚖️ **Score Fusion**: Combined all signals with learned weights",
                "🎯 **Final Ranking**: Applied diversity and novelty constraints"
            ]
            
            for step in reasoning_steps:
                st.markdown(step)
                st.progress(np.random.uniform(0.7, 0.95))
    
    def render_adaptation_tab(self):
        """Render real-time adaptation tab"""
        st.markdown("## ⚡ Real-Time Adaptation Dashboard")
        
        # Adaptation controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            adaptation_enabled = st.checkbox("Enable Real-time Adaptation", value=True)
        
        with col2:
            adaptation_sensitivity = st.slider(
                "Adaptation Sensitivity",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        with col3:
            feedback_window = st.selectbox(
                "Feedback Window",
                options=[10, 25, 50, 100],
                index=1
            )
        
        # Live feedback processing
        st.markdown("### 📊 Live Feedback Processing")
        
        if st.button("Simulate Feedback Batch"):
            with st.spinner("Processing feedback batch..."):
                time.sleep(2)  # Simulate processing
                st.success("Processed 23 feedback items • Model updated • Cache invalidated")
        
        # Adaptation visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Preference drift over time
            drift_data = self.generate_preference_drift_data()
            
            fig = go.Figure()
            
            for genre in ['Action', 'Comedy', 'Drama', 'Horror']:
                fig.add_trace(go.Scatter(
                    x=drift_data['timestamp'],
                    y=drift_data[genre.lower()],
                    mode='lines+markers',
                    name=genre
                ))
            
            fig.update_layout(
                title="Preference Drift Over Time",
                xaxis_title="Time",
                yaxis_title="Preference Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Adaptation triggers
            trigger_data = {
                'Trigger Type': ['Concept Drift', 'Preference Shift', 'Feedback Surge', 'Performance Drop'],
                'Frequency': [12, 8, 15, 3],
                'Severity': ['Medium', 'High', 'Low', 'Critical']
            }
            
            fig = px.bar(
                x=trigger_data['Trigger Type'],
                y=trigger_data['Frequency'],
                color=trigger_data['Severity'],
                title="Adaptation Triggers (Last 24h)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Before/After comparison
        st.markdown("### 🔄 Before/After Adaptation Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Before Adaptation")
            before_recs = self.demo_data.generate_sample_recommendations('movies', 3)
            for rec in before_recs:
                st.markdown(f"- **{rec['title']}** (Score: {rec['score']:.2f})")
        
        with col2:
            st.markdown("#### After Adaptation")
            after_recs = self.demo_data.generate_sample_recommendations('movies', 3)
            for rec in after_recs:
                st.markdown(f"- **{rec['title']}** (Score: {rec['score']:.2f})")
        
        # Adaptation metrics
        st.markdown("### 📈 Adaptation Impact Metrics")
        
        adaptation_metrics = pd.DataFrame({
            'Metric': ['NDCG@10', 'Diversity', 'User Satisfaction', 'Click-through Rate'],
            'Before': [0.342, 0.684, 3.2, 0.084],
            'After': [0.378, 0.721, 3.7, 0.096],
            'Improvement': ['+10.5%', '+5.4%', '+15.6%', '+14.3%']
        })
        
        st.dataframe(adaptation_metrics, use_container_width=True)
    
    def render_testing_tab(self):
        """Render A/B testing tab"""
        st.markdown("## 🧪 A/B Testing & Experimentation")
        
        # Active experiments
        st.markdown("### 🔬 Active Experiments")
        
        experiments = [
            {
                'name': 'Hybrid Weight Optimization',
                'status': 'Running',
                'traffic': '50%',
                'duration': '7 days',
                'primary_metric': 'NDCG@10',
                'current_lift': '+2.3%',
                'significance': '87%'
            },
            {
                'name': 'Diversity Algorithm V2',
                'status': 'Running',
                'traffic': '25%',
                'duration': '5 days',
                'primary_metric': 'Diversity Score',
                'current_lift': '+8.1%',
                'significance': '94%'
            },
            {
                'name': 'Cold Start Improvement',
                'status': 'Ramping',
                'traffic': '10%',
                'duration': '2 days',
                'primary_metric': 'Coverage',
                'current_lift': '+1.2%',
                'significance': '23%'
            }
        ]
        
        for exp in experiments:
            with st.expander(f"🧪 {exp['name']} - {exp['status']}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Traffic Split", exp['traffic'])
                    st.metric("Duration", exp['duration'])
                
                with col2:
                    st.metric("Primary Metric", exp['primary_metric'])
                    st.metric("Current Lift", exp['current_lift'])
                
                with col3:
                    significance = float(exp['significance'].strip('%'))
                    color = "🟢" if significance > 95 else "🟡" if significance > 80 else "🔴"
                    st.metric(f"{color} Significance", exp['significance'])
                
                with col4:
                    status_color = "🟢" if exp['status'] == 'Running' else "🟡"
                    st.metric(f"{status_color} Status", exp['status'])
                
                # Quick actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"View Details", key=f"details_{exp['name']}"):
                        st.info("Opening detailed experiment analysis...")
                with col2:
                    if st.button(f"Pause", key=f"pause_{exp['name']}"):
                        st.warning("Experiment paused")
                with col3:
                    if st.button(f"Stop", key=f"stop_{exp['name']}"):
                        st.error("Experiment stopped")
        
        # Experiment results visualization
        st.markdown("### 📊 Experiment Results")
        
        # Sample A/B test results
        test_results = pd.DataFrame({
            'Variant': ['Control', 'Treatment A', 'Treatment B'],
            'Users': [5000, 5000, 2500],
            'NDCG@10': [0.342, 0.357, 0.351],
            'Diversity': [0.684, 0.721, 0.698],
            'CTR': [0.084, 0.091, 0.087],
            'Significance': ['-', '95.2%', '78.3%']
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                test_results,
                x='Variant',
                y='NDCG@10',
                title="NDCG@10 by Variant",
                color='Variant'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                test_results,
                x='Variant',
                y='Diversity',
                title="Diversity by Variant",
                color='Variant'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical significance testing
        st.markdown("### 📈 Statistical Analysis")
        
        with st.expander("Statistical Test Details"):
            st.markdown("""
            **Test Type:** Welch's t-test (unequal variances)
            
            **Control vs Treatment A:**
            - t-statistic: 3.47
            - p-value: 0.0005
            - 95% CI: [0.008, 0.022]
            - Effect size (Cohen's d): 0.18
            
            **Recommendation:** ✅ Declare Treatment A as winner
            """)
        
        # Create new experiment
        st.markdown("### ➕ Create New Experiment")
        
        with st.expander("New Experiment Configuration"):
            exp_name = st.text_input("Experiment Name")
            exp_description = st.text_area("Description")
            
            col1, col2 = st.columns(2)
            with col1:
                traffic_split = st.slider("Traffic Split (%)", 5, 50, 25)
                duration_days = st.number_input("Duration (days)", 1, 30, 14)
            
            with col2:
                primary_metric = st.selectbox(
                    "Primary Metric",
                    ["NDCG@10", "Diversity", "CTR", "User Satisfaction"]
                )
                minimum_effect = st.number_input("Minimum Detectable Effect (%)", 1.0, 20.0, 5.0)
            
            if st.button("Create Experiment"):
                st.success(f"Experiment '{exp_name}' created successfully!")
        
        # Load testing section
        st.markdown("### ⚡ Performance Testing")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            concurrent_users = st.number_input("Concurrent Users", 10, 1000, 100)
        
        with col2:
            test_duration = st.number_input("Duration (seconds)", 30, 600, 60)
        
        with col3:
            if st.button("Run Load Test", type="primary"):
                with st.spinner("Running load test..."):
                    # Simulate load test
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                    
                    # Show results
                    st.success("Load test completed!")
                    
                    results = {
                        'Total Requests': 5420,
                        'Requests/sec': 90.3,
                        'Avg Response Time': '87ms',
                        'P95 Response Time': '156ms',
                        'Error Rate': '0.2%',
                        'Performance Grade': 'A'
                    }
                    
                    cols = st.columns(len(results))
                    for i, (metric, value) in enumerate(results.items()):
                        with cols[i]:
                            st.metric(metric, value)
    
    def render_fairness_tab(self):
        """Render fairness monitoring tab"""
        st.markdown("## ⚖️ Fairness & Bias Monitoring")
        st.markdown("*Ensuring equitable recommendations across all user groups*")
        
        # Fairness overview metrics
        st.markdown("### 📊 Fairness Metrics Overview")
        
        fairness_data = self.generate_fairness_metrics()
        
        # Key fairness indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            parity_score = fairness_data['demographic_parity']
            color = "🟢" if parity_score > 0.8 else "🟡" if parity_score > 0.6 else "🔴"
            st.metric(
                "Demographic Parity",
                f"{parity_score:.3f}",
                delta=f"{parity_score - 0.8:.3f}",
                help="Measures equal opportunity across demographic groups"
            )
            st.markdown(f"{color} {'Excellent' if parity_score > 0.8 else 'Good' if parity_score > 0.6 else 'Needs Attention'}")
        
        with col2:
            representation_score = fairness_data['representation']
            color = "🟢" if representation_score > 0.85 else "🟡" if representation_score > 0.7 else "🔴"
            st.metric(
                "Content Representation",
                f"{representation_score:.3f}",
                delta=f"{representation_score - 0.85:.3f}",
                help="Measures diversity in recommended content across groups"
            )
            st.markdown(f"{color} {'Balanced' if representation_score > 0.85 else 'Moderate' if representation_score > 0.7 else 'Imbalanced'}")
        
        with col3:
            group_fairness = fairness_data['group_fairness']
            color = "🟢" if group_fairness > 0.8 else "🟡" if group_fairness > 0.6 else "🔴"
            st.metric(
                "Group Fairness",
                f"{group_fairness:.3f}",
                delta=f"{group_fairness - 0.8:.3f}",
                help="Measures recommendation quality equality across groups"
            )
            st.markdown(f"{color} {'Fair' if group_fairness > 0.8 else 'Moderate' if group_fairness > 0.6 else 'Unfair'}")
        
        with col4:
            overall_fairness = np.mean([parity_score, representation_score, group_fairness])
            color = "🟢" if overall_fairness > 0.8 else "🟡" if overall_fairness > 0.6 else "🔴"
            st.metric(
                "Overall Fairness",
                f"{overall_fairness:.3f}",
                delta=f"{overall_fairness - 0.8:.3f}",
                help="Combined fairness score across all metrics"
            )
            st.markdown(f"{color} Overall Rating")
        
        # Demographic group analysis
        st.markdown("### 👥 Demographic Group Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution chart
            gender_data = fairness_data['demographic_breakdown']['gender']
            
            fig_gender = px.bar(
                x=list(gender_data.keys()),
                y=list(gender_data.values()),
                title="Recommendation Quality by Gender",
                labels={'x': 'Gender', 'y': 'Average Score'},
                color=list(gender_data.values()),
                color_continuous_scale='RdYlGn'
            )
            fig_gender.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            # Age group distribution chart
            age_data = fairness_data['demographic_breakdown']['age_group']
            
            fig_age = px.bar(
                x=list(age_data.keys()),
                y=list(age_data.values()),
                title="Recommendation Quality by Age Group",
                labels={'x': 'Age Group', 'y': 'Average Score'},
                color=list(age_data.values()),
                color_continuous_scale='RdYlGn'
            )
            fig_age.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Content representation analysis
        st.markdown("### 🎭 Content Representation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre representation by gender
            genre_gender_data = fairness_data['content_representation']['genre_by_gender']
            
            fig_genre_gender = px.imshow(
                genre_gender_data,
                title="Genre Recommendations by Gender",
                labels=dict(x="Gender", y="Genre", color="Recommendation Rate"),
                color_continuous_scale='RdYlGn'
            )
            fig_genre_gender.update_layout(height=400)
            st.plotly_chart(fig_genre_gender, use_container_width=True)
        
        with col2:
            # Content type distribution
            content_dist = fairness_data['content_representation']['content_types']
            
            fig_content = px.pie(
                values=list(content_dist.values()),
                names=list(content_dist.keys()),
                title="Content Type Distribution"
            )
            fig_content.update_layout(height=400)
            st.plotly_chart(fig_content, use_container_width=True)
        
        # Bias detection alerts
        st.markdown("### 🚨 Bias Detection & Alerts")
        
        alerts = fairness_data.get('alerts', [])
        
        if alerts:
            for alert in alerts:
                severity_color = {
                    'low': 'info',
                    'medium': 'warning', 
                    'high': 'error',
                    'critical': 'error'
                }
                
                with st.container():
                    if alert['severity'] == 'critical':
                        st.error(f"🚨 CRITICAL: {alert['message']}")
                    elif alert['severity'] == 'high':
                        st.error(f"⚠️ HIGH: {alert['message']}")
                    elif alert['severity'] == 'medium':
                        st.warning(f"🔶 MEDIUM: {alert['message']}")
                    else:
                        st.info(f"ℹ️ LOW: {alert['message']}")
                    
                    # Show recommendations for fixing the issue
                    if alert.get('recommendations'):
                        with st.expander("💡 Recommendations to Address This Issue"):
                            for i, rec in enumerate(alert['recommendations'], 1):
                                st.markdown(f"{i}. {rec}")
        else:
            st.success("✅ No significant bias detected in current recommendations!")
        
        # Fairness trends over time
        st.markdown("### 📈 Fairness Trends")
        
        fairness_trends = self.generate_fairness_trends()
        
        fig_trends = go.Figure()
        
        fig_trends.add_trace(go.Scatter(
            x=fairness_trends['timestamps'],
            y=fairness_trends['demographic_parity'],
            mode='lines+markers',
            name='Demographic Parity',
            line=dict(color='blue')
        ))
        
        fig_trends.add_trace(go.Scatter(
            x=fairness_trends['timestamps'],
            y=fairness_trends['representation'],
            mode='lines+markers',
            name='Representation',
            line=dict(color='green')
        ))
        
        fig_trends.add_trace(go.Scatter(
            x=fairness_trends['timestamps'],
            y=fairness_trends['group_fairness'],
            mode='lines+markers',
            name='Group Fairness',
            line=dict(color='red')
        ))
        
        # Add threshold line
        fig_trends.add_hline(y=0.8, line_dash="dash", line_color="gray", 
                            annotation_text="Fairness Threshold (0.8)")
        
        fig_trends.update_layout(
            title="Fairness Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Fairness Score",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Mitigation strategies
        st.markdown("### 🛠️ Bias Mitigation Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Active Strategies")
            strategies = [
                "✅ Demographic-aware recommendation balancing",
                "✅ Content diversity enforcement",
                "✅ Real-time fairness monitoring",
                "✅ User feedback incorporation",
                "✅ Algorithmic bias testing"
            ]
            for strategy in strategies:
                st.markdown(strategy)
        
        with col2:
            st.markdown("#### 📋 Recommended Actions")
            actions = fairness_data.get('recommended_actions', [
                "Continue monitoring current fairness levels",
                "Regular bias audit every 30 days",
                "A/B test fairness-aware algorithms",
                "Expand content diversity in catalog",
                "Collect more diverse user feedback"
            ])
            for i, action in enumerate(actions, 1):
                st.markdown(f"{i}. {action}")
        
        # Compliance status
        st.markdown("### 📋 Compliance Status")
        
        compliance_status = fairness_data.get('compliance_status', 'good')
        
        if compliance_status == 'excellent':
            st.success("🏆 Excellent - Exceeds fairness standards across all metrics")
        elif compliance_status == 'good':
            st.success("✅ Good - Meets fairness standards with minor areas for improvement")
        elif compliance_status == 'acceptable':
            st.warning("⚠️ Acceptable - Meets minimum standards but requires attention")
        else:
            st.error("🚨 Needs Improvement - Below acceptable fairness thresholds")
    
    def generate_recommendations(self, user_id: str) -> Dict[str, List[Dict]]:
        """Generate sample recommendations"""
        try:
            # Try to call actual API
            response = requests.post(
                f"{self.api_base_url}/api/v1/recommendations",
                json={
                    "user_id": user_id,
                    "content_types": ["movies", "books", "music"],
                    "num_recommendations": 6
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        # Fallback to demo data
        return {
            'movies': self.demo_data.generate_sample_recommendations('movies', 6),
            'books': self.demo_data.generate_sample_recommendations('books', 6),
            'music': self.demo_data.generate_sample_recommendations('music', 6)
        }
    
    def record_feedback(self, item_id: str, rating: int, feedback_type: str):
        """Record user feedback"""
        feedback = {
            'item_id': item_id,
            'rating': rating,
            'feedback_type': feedback_type,
            'timestamp': datetime.now().isoformat(),
            'user_id': st.session_state.user_preferences.get('user_id', 'demo_user')
        }
        
        st.session_state.feedback_history.append(feedback)
        
        # Try to send to API
        try:
            requests.post(
                f"{self.api_base_url}/api/v1/feedback",
                json=feedback,
                timeout=5
            )
        except:
            pass  # Graceful fallback for demo
    
    def generate_sample_metrics(self) -> Dict[str, float]:
        """Generate realistic sample metrics"""
        base_time = time.time()
        variation = np.sin(base_time / 3600) * 0.05  # Hourly variation
        
        return {
            'ndcg_at_10': max(0.1, min(0.9, 0.37 + variation + np.random.normal(0, 0.02))),
            'diversity': max(0.3, min(1.0, 0.73 + variation + np.random.normal(0, 0.03))),
            'avg_latency_ms': max(20, 85 + np.random.exponential(15)),
            'throughput_rps': max(10, 750 + np.random.normal(0, 100))
        }
    
    def generate_time_series_data(self) -> Dict[str, List]:
        """Generate time series data for charts"""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='H'
        )
        
        ndcg_base = 0.37
        diversity_base = 0.73
        
        ndcg_values = []
        diversity_values = []
        
        for i, ts in enumerate(timestamps):
            hour_factor = np.sin(i * 2 * np.pi / 24) * 0.02  # Daily pattern
            noise = np.random.normal(0, 0.01)
            
            ndcg_values.append(max(0.1, min(0.9, ndcg_base + hour_factor + noise)))
            diversity_values.append(max(0.3, min(1.0, diversity_base + hour_factor * 0.5 + noise)))
        
        return {
            'timestamp': timestamps,
            'ndcg': ndcg_values,
            'diversity': diversity_values
        }
    
    def generate_preference_drift_data(self) -> Dict[str, List]:
        """Generate preference drift data"""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='D'
        )
        
        data = {'timestamp': timestamps}
        
        for genre in ['action', 'comedy', 'drama', 'horror']:
            # Simulate preference drift
            values = []
            base_value = np.random.uniform(0.3, 0.7)
            
            for i in range(len(timestamps)):
                trend = (i / len(timestamps)) * np.random.uniform(-0.2, 0.2)
                noise = np.random.normal(0, 0.05)
                value = max(0, min(1, base_value + trend + noise))
                values.append(value)
            
            data[genre] = values
        
        return data
    
    def generate_explanation(self, item: Dict[str, Any], content_type: str) -> str:
        """Generate AI explanation for recommendation"""
        explanations = [
            f"🎯 **Why this {content_type}?** Based on your preference for {item.get('genre', 'this genre')} "
            f"content and your recent interest in {item.get('mood', 'similar themes')}, this item aligns "
            f"perfectly with your taste profile.",
            
            f"👥 **Community Insight:** Users with similar preferences rated this {item.get('score', 0.8):.1f}/1.0. "
            f"The collaborative filtering algorithm identified strong patterns among users who enjoyed "
            f"both this item and content you've previously rated highly.",
            
            f"🧠 **Content Analysis:** Our AI analyzed the thematic elements, narrative structure, and "
            f"stylistic features of this {content_type}. It shares key characteristics with your "
            f"top-rated items, particularly in {item.get('genre', 'genre')} storytelling.",
            
            f"🔍 **Knowledge Graph Connection:** This recommendation connects to your interests through "
            f"our knowledge graph, which identified relationships between {item.get('genre', 'your preferences')}, "
            f"creative influences, and thematic elements you enjoy."
        ]
        
        return "\n\n".join(explanations)
    
    def generate_similar_users(self) -> List[Dict[str, Any]]:
        """Generate similar users data"""
        return [
            {
                'name': 'Alex Chen',
                'similarity': 0.89,
                'common_interests': 'Sci-Fi movies, Mystery novels, Electronic music'
            },
            {
                'name': 'Maria Rodriguez',
                'similarity': 0.82,
                'common_interests': 'Action thrillers, Biography books, Rock music'
            },
            {
                'name': 'David Kim',
                'similarity': 0.76,
                'common_interests': 'Drama series, Philosophy books, Jazz music'
            }
        ]
    
    def generate_similar_items(self, content_type: str) -> List[Dict[str, Any]]:
        """Generate similar items data"""
        if content_type == 'movies':
            return [
                {'title': 'Blade Runner 2049', 'similarity': 0.91, 'reason': 'Shared sci-fi themes and visual style'},
                {'title': 'Interstellar', 'similarity': 0.87, 'reason': 'Similar emotional depth and scientific concepts'},
                {'title': 'Ex Machina', 'similarity': 0.83, 'reason': 'AI themes and philosophical questions'}
            ]
        elif content_type == 'books':
            return [
                {'title': 'Foundation', 'similarity': 0.89, 'reason': 'Epic sci-fi scope and world-building'},
                {'title': 'Dune', 'similarity': 0.85, 'reason': 'Complex political intrigue and unique universe'},
                {'title': 'Hyperion', 'similarity': 0.81, 'reason': 'Literary sci-fi with multiple narratives'}
            ]
        else:  # music
            return [
                {'title': 'Synthwave Collection', 'similarity': 0.88, 'reason': 'Electronic soundscapes and atmosphere'},
                {'title': 'Ambient Space', 'similarity': 0.84, 'reason': 'Cosmic themes and ethereal production'},
                {'title': 'Cyberpunk Beats', 'similarity': 0.79, 'reason': 'Futuristic sound design and energy'}
            ]
    
    def generate_fairness_metrics(self) -> Dict[str, Any]:
        """Generate sample fairness metrics data"""
        return {
            'demographic_parity': 0.83,
            'representation': 0.78,
            'group_fairness': 0.85,
            'demographic_breakdown': {
                'gender': {
                    'Male': 0.82,
                    'Female': 0.79,
                    'Non-binary': 0.76,
                    'Prefer not to say': 0.81
                },
                'age_group': {
                    '18-24': 0.84,
                    '25-34': 0.86,
                    '35-44': 0.81,
                    '45-54': 0.77,
                    '55+': 0.73
                }
            },
            'content_representation': {
                'genre_by_gender': [
                    [0.8, 0.7, 0.6],  # Action by gender
                    [0.6, 0.9, 0.8],  # Romance by gender
                    [0.7, 0.8, 0.7],  # Drama by gender
                    [0.9, 0.5, 0.7],  # Horror by gender
                ],
                'content_types': {
                    'Movies': 0.4,
                    'TV Shows': 0.3,
                    'Books': 0.2,
                    'Music': 0.1
                }
            },
            'alerts': [
                {
                    'severity': 'medium',
                    'message': 'Slight underrepresentation of horror content for female users',
                    'recommendations': [
                        'Review horror content recommendations for gender bias',
                        'Implement content balancing for underrepresented genres',
                        'Collect more diverse user feedback on horror preferences'
                    ]
                }
            ],
            'compliance_status': 'good',
            'recommended_actions': [
                'Continue monitoring current fairness levels',
                'Regular bias audit every 30 days',
                'A/B test fairness-aware algorithms',
                'Expand content diversity in catalog',
                'Collect more diverse user feedback'
            ]
        }
    
    def generate_fairness_trends(self) -> Dict[str, List]:
        """Generate fairness trends over time"""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='H'
        )
        
        base_parity = 0.83
        base_representation = 0.78
        base_group_fairness = 0.85
        
        parity_values = []
        representation_values = []
        group_fairness_values = []
        
        for i, ts in enumerate(timestamps):
            # Simulate daily and random variations
            daily_factor = np.sin(i * 2 * np.pi / 24) * 0.02
            noise = np.random.normal(0, 0.01)
            
            parity_values.append(max(0.5, min(1.0, base_parity + daily_factor + noise)))
            representation_values.append(max(0.5, min(1.0, base_representation + daily_factor * 0.8 + noise)))
            group_fairness_values.append(max(0.5, min(1.0, base_group_fairness + daily_factor * 0.6 + noise)))
        
        return {
            'timestamps': timestamps,
            'demographic_parity': parity_values,
            'representation': representation_values,
            'group_fairness': group_fairness_values
        }
    
    def run(self):
        """Main method to run the Streamlit application"""
        # Render header
        self.render_header()
        
        # Render sidebar and get user selection
        selected_user, user_data = self.render_sidebar()
        
        # Render main dashboard with tabs if user is selected
        if selected_user and user_data:
            self.render_main_dashboard(selected_user, user_data)
        else:
            st.warning("Please select a user from the sidebar to continue.")

# Main execution
def main():
    """Main function to run the Streamlit app"""
    try:
        app = DemoApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
else:
    # For streamlit run command
    app = DemoApp()
    app.run()