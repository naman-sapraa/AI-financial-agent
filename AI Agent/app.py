import os
import time
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import schedule
import threading
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ta
import yfinance as yf
import tweepy
import logging
from dotenv import load_dotenv
from newspaper import Article
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FinancialAgent")

# Download NLTK resources for sentiment analysis
nltk.download('vader_lexicon', quiet=True)

class IndianFinancialMarketAgent:
    def __init__(self):
        # API Keys (store these in environment variables)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_secret = os.getenv('TWITTER_ACCESS_SECRET')
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Data storage
        self.market_data = {}
        self.news_data = []
        self.social_media_data = []
        self.economic_indicators = {}
        self.analysis_results = {}
        
        # Major Indian market indices and their constituents
        self.indices = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY BANK': '^NSEBANK',
            'NIFTY IT': '^CNXIT',
            'NIFTY PHARMA': '^CNXPHARMA'
        }
        
        # Top stocks to track (example list - can be expanded)
        self.stocks = [
            'RELIANCE.NS',    # Reliance Industries
            'TCS.NS',         # Tata Consultancy Services
            'HDFCBANK.NS',    # HDFC Bank
            'INFY.NS',        # Infosys
            'HINDUNILVR.NS',  # Hindustan Unilever
            'ICICIBANK.NS',   # ICICI Bank
            'SBIN.NS',        # State Bank of India
            'BAJFINANCE.NS',  # Bajaj Finance
            'BHARTIARTL.NS',  # Bharti Airtel
            'ITC.NS',         # ITC Limited
            'KOTAKBANK.NS',   # Kotak Mahindra Bank
            'ADANIPORTS.NS',  # Adani Ports
            'ASIANPAINT.NS',  # Asian Paints
            'AXISBANK.NS',    # Axis Bank
            'TATAMOTORS.NS'   # Tata Motors
        ]
        
        # Define sectors for analysis
        self.sectors = {
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
            'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
            'Consumer': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'MARICO.NS'],
            'Energy': ['RELIANCE.NS', 'ONGC.NS', 'POWERGRID.NS', 'NTPC.NS', 'BPCL.NS'],
            'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'AUROPHARMA.NS']
        }
        
        # Initialize the database (using pandas DataFrames for simplicity)
        self.initialize_database()
        
        # Initialize technical indicators
        self.technical_indicators = [
            'rsi', 'macd', 'bollinger_bands', 'moving_averages'
        ]
        
        logger.info("Indian Financial Market Agent initialized")
    
    def initialize_database(self):
        """Initialize the historical database with empty DataFrames"""
        self.historical_prices = {}
        self.historical_news = pd.DataFrame(columns=['date', 'source', 'title', 'content', 'sentiment'])
        self.historical_social = pd.DataFrame(columns=['date', 'platform', 'content', 'sentiment'])
        self.historical_indicators = pd.DataFrame()
        
        # Create directory for data storage if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        logger.info("Database initialized")
    
    def fetch_market_data(self):
        """Fetch current market data for indices and stocks"""
        logger.info("Fetching market data...")
        
        # Fetch data for indices
        for index_name, index_symbol in self.indices.items():
            try:
                data = yf.download(index_symbol, period="5d", interval="1d")
                if not data.empty:
                    self.market_data[index_name] = {
                        'current': data['Close'].iloc[-1],
                        'previous': data['Close'].iloc[-2],
                        'change': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100,
                        'volume': data['Volume'].iloc[-1],
                        'data': data
                    }
                    logger.info(f"Fetched {index_name} data: {self.market_data[index_name]['current']}")
            except Exception as e:
                logger.error(f"Error fetching {index_name} data: {str(e)}")
        
        # Fetch data for individual stocks
        for stock in self.stocks:
            try:
                data = yf.download(stock, period="5d", interval="1d")
                if not data.empty:
                    self.market_data[stock] = {
                        'current': data['Close'].iloc[-1],
                        'previous': data['Close'].iloc[-2],
                        'change': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100,
                        'volume': data['Volume'].iloc[-1],
                        'data': data
                    }
                    # Store in historical database
                    self.historical_prices[stock] = data if stock not in self.historical_prices else pd.concat([self.historical_prices[stock], data]).drop_duplicates()
            except Exception as e:
                logger.error(f"Error fetching {stock} data: {str(e)}")
        
        logger.info("Market data fetched successfully")
        return self.market_data
    
    def fetch_news_data(self):
        """Fetch financial news from various sources"""
        logger.info("Fetching news data...")
        
        # Sources for Indian financial news
        sources = [
            'economictimes.indiatimes.com',
            'moneycontrol.com',
            'livemint.com',
            'financialexpress.com',
            'business-standard.com'
        ]
        
        # Use News API to get recent financial news
        try:
            url = f"https://newsapi.org/v2/everything?q=indian%20stock%20market%20OR%20sensex%20OR%20nifty&domains={','.join(sources)}&language=en&sortBy=publishedAt&apiKey={self.news_api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                news_items = response.json().get('articles', [])
                
                for item in news_items[:20]:  # Limit to 20 most recent articles
                    # Extract main content using newspaper3k
                    try:
                        article = Article(item['url'])
                        article.download()
                        article.parse()
                        content = article.text
                    except:
                        content = item.get('description', '')
                    
                    # Analyze sentiment
                    sentiment_score = self.sentiment_analyzer.polarity_scores(content)
                    
                    news_item = {
                        'date': item.get('publishedAt', ''),
                        'source': item.get('source', {}).get('name', ''),
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'content': content,
                        'sentiment': sentiment_score
                    }
                    
                    self.news_data.append(news_item)
                
                # Store in historical database
                news_df = pd.DataFrame(self.news_data)
                self.historical_news = pd.concat([self.historical_news, news_df]).drop_duplicates(subset=['title', 'source'])
                
                logger.info(f"Fetched {len(news_items)} news articles")
            else:
                logger.error(f"Failed to fetch news: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
        
        return self.news_data
    
    def fetch_social_media_data(self):
        """Fetch social media sentiment from Twitter/X"""
        logger.info("Fetching social media data...")
        
        try:
            # Set up Tweepy client
            auth = tweepy.OAuth1UserHandler(
                self.twitter_api_key, self.twitter_api_secret,
                self.twitter_access_token, self.twitter_access_secret
            )
            api = tweepy.API(auth)
            
            # Search terms for Indian market
            search_terms = [
                "NIFTY", "Sensex", "NSE India", "BSE India", 
                "Indian stocks", "Indian market"
            ]
            
            social_data = []
            
            for term in search_terms:
                tweets = api.search_tweets(q=term, lang="en", count=50, tweet_mode="extended")
                
                for tweet in tweets:
                    content = tweet.full_text
                    sentiment_score = self.sentiment_analyzer.polarity_scores(content)
                    
                    tweet_data = {
                        'date': tweet.created_at,
                        'platform': 'Twitter',
                        'content': content,
                        'sentiment': sentiment_score,
                        'search_term': term
                    }
                    
                    social_data.append(tweet_data)
            
            self.social_media_data = social_data
            
            # Store in historical database
            social_df = pd.DataFrame(social_data)
            self.historical_social = pd.concat([self.historical_social, social_df]).drop_duplicates(subset=['content'])
            
            logger.info(f"Fetched {len(social_data)} social media posts")
            
        except Exception as e:
            logger.error(f"Error fetching social media data: {str(e)}")
        
        return self.social_media_data
    
    def fetch_economic_indicators(self):
        """Fetch key Indian economic indicators"""
        logger.info("Fetching economic indicators...")
        
        # In a real implementation, you would use an API for this data
        # For demonstration, we'll use some sample data
        
        self.economic_indicators = {
            'GDP_Growth': 6.3,  # Example value
            'Inflation_Rate': 4.7,  # Example value
            'Repo_Rate': 6.5,  # Example value
            'Unemployment': 7.8,  # Example value
            'Foreign_Exchange_Reserves': 642.5,  # Example value in USD billions
            'INR_USD': 82.94,  # Example value
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # In a real implementation, append to historical database
        logger.info("Economic indicators fetched")
        
        return self.economic_indicators
    
    def perform_technical_analysis(self):
        """Perform technical analysis on stock price data"""
        logger.info("Performing technical analysis...")
        
        technical_results = {}
        
        for stock, data in self.historical_prices.items():
            if data.empty:
                continue
                
            # Get the last 100 days of data if available
            df = data.tail(100).copy()
            
            if len(df) < 14:  # Need at least 14 days for most indicators
                continue
                
            # Calculate RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            # Calculate MACD
            macd = ta.trend.MACD(df['Close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Calculate Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['bollinger_high'] = bollinger.bollinger_hband()
            df['bollinger_low'] = bollinger.bollinger_lband()
            df['bollinger_mid'] = bollinger.bollinger_mavg()
            
            # Calculate Moving Averages
            df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            df['sma_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
            
            # Get latest values for analysis
            latest = df.iloc[-1]
            
            technical_results[stock] = {
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'macd_diff': latest['macd_diff'],
                'bollinger_high': latest['bollinger_high'],
                'bollinger_low': latest['bollinger_low'],
                'bollinger_mid': latest['bollinger_mid'],
                'sma_20': latest['sma_20'],
                'sma_50': latest['sma_50'],
                'sma_200': latest['sma_200'] if 'sma_200' in latest else None,
                'price': latest['Close'],
                'volume': latest['Volume'],
                'signal': self.generate_signal(latest)
            }
        
        self.analysis_results['technical'] = technical_results
        logger.info("Technical analysis completed")
        
        return technical_results
    
    def generate_signal(self, data):
        """Generate trading signals based on technical indicators"""
        signals = []
        
        # RSI conditions
        if 'rsi' in data and not pd.isna(data['rsi']):
            if data['rsi'] < 30:
                signals.append('RSI Oversold')
            elif data['rsi'] > 70:
                signals.append('RSI Overbought')
        
        # MACD conditions
        if 'macd' in data and 'macd_signal' in data and not pd.isna(data['macd']) and not pd.isna(data['macd_signal']):
            if data['macd'] > data['macd_signal'] and data['macd_diff'] > 0:
                signals.append('MACD Bullish')
            elif data['macd'] < data['macd_signal'] and data['macd_diff'] < 0:
                signals.append('MACD Bearish')
        
        # Bollinger Bands conditions
        if 'bollinger_high' in data and 'bollinger_low' in data and 'Close' in data:
            if data['Close'] > data['bollinger_high']:
                signals.append('Price Above Upper Bollinger')
            elif data['Close'] < data['bollinger_low']:
                signals.append('Price Below Lower Bollinger')
        
        # Moving Average conditions
        if 'sma_20' in data and 'sma_50' in data and 'Close' in data:
            if data['Close'] > data['sma_20'] and data['sma_20'] > data['sma_50']:
                signals.append('Bullish MA Alignment')
            elif data['Close'] < data['sma_20'] and data['sma_20'] < data['sma_50']:
                signals.append('Bearish MA Alignment')
            
            if 'sma_200' in data and not pd.isna(data['sma_200']):
                if data['Close'] > data['sma_200']:
                    signals.append('Above 200 SMA')
                else:
                    signals.append('Below 200 SMA')
        
        return signals
    
    def analyze_sentiment(self):
        """Analyze sentiment from news and social media"""
        logger.info("Analyzing sentiment...")
        
        sentiment_results = {
            'overall': 0,
            'news': 0,
            'social': 0,
            'by_sector': {},
            'by_stock': {},
            'narratives': []
        }
        
        # Calculate overall news sentiment
        if self.news_data:
            news_sentiment = [item['sentiment']['compound'] for item in self.news_data]
            sentiment_results['news'] = sum(news_sentiment) / len(news_sentiment)
            
            # Extract key narratives
            for item in sorted(self.news_data, key=lambda x: abs(x['sentiment']['compound']), reverse=True)[:5]:
                sentiment_results['narratives'].append({
                    'title': item['title'],
                    'sentiment': item['sentiment']['compound'],
                    'source': item['source']
                })
        
        # Calculate overall social media sentiment
        if self.social_media_data:
            social_sentiment = [item['sentiment']['compound'] for item in self.social_media_data]
            sentiment_results['social'] = sum(social_sentiment) / len(social_sentiment)
        
        # Combine news and social for overall sentiment
        if sentiment_results['news'] != 0 or sentiment_results['social'] != 0:
            sentiment_results['overall'] = (sentiment_results['news'] + sentiment_results['social']) / 2
        
        # Analyze sentiment by sector
        for sector, stocks in self.sectors.items():
            sector_news = []
            
            # Find news related to stocks in this sector
            for item in self.news_data:
                content = item['title'] + ' ' + item['content']
                if any(stock.split('.')[0] in content for stock in stocks):
                    sector_news.append(item['sentiment']['compound'])
            
            if sector_news:
                sentiment_results['by_sector'][sector] = sum(sector_news) / len(sector_news)
            else:
                sentiment_results['by_sector'][sector] = 0
        
        # Analyze sentiment by stock
        for stock in self.stocks:
            stock_name = stock.split('.')[0]
            stock_news = []
            
            # Find news related to this stock
            for item in self.news_data:
                content = item['title'] + ' ' + item['content']
                if stock_name in content:
                    stock_news.append(item['sentiment']['compound'])
            
            if stock_news:
                sentiment_results['by_stock'][stock] = sum(stock_news) / len(stock_news)
            else:
                sentiment_results['by_stock'][stock] = 0
        
        self.analysis_results['sentiment'] = sentiment_results
        logger.info("Sentiment analysis completed")
        
        return sentiment_results
    
    def perform_correlation_analysis(self):
        """Analyze correlations between different assets"""
        logger.info("Performing correlation analysis...")
        
        # Need at least 2 stocks with data for correlation
        if len(self.historical_prices) < 2:
            logger.warning("Not enough historical data for correlation analysis")
            return {}
        
        # Create a DataFrame with closing prices for all stocks
        prices_df = pd.DataFrame()
        
        for stock, data in self.historical_prices.items():
            if not data.empty:
                prices_df[stock] = data['Close']
        
        # Calculate correlation matrix
        correlation_matrix = prices_df.corr()
        
        # Find highly correlated and inversely correlated pairs
        high_corr_pairs = []
        inverse_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                stock1 = correlation_matrix.columns[i]
                stock2 = correlation_matrix.columns[j]
                corr = correlation_matrix.iloc[i, j]
                
                if corr > 0.8:
                    high_corr_pairs.append((stock1, stock2, corr))
                elif corr < -0.5:
                    inverse_corr_pairs.append((stock1, stock2, corr))
        
        correlation_results = {
            'matrix': correlation_matrix.to_dict(),
            'high_correlation': high_corr_pairs,
            'inverse_correlation': inverse_corr_pairs
        }
        
        self.analysis_results['correlation'] = correlation_results
        logger.info("Correlation analysis completed")
        
        return correlation_results
    
    def identify_outliers(self):
        """Identify outlier performers in the market"""
        logger.info("Identifying market outliers...")
        
        outliers = {
            'positive': [],
            'negative': [],
            'volume': [],
            'volatility': []
        }
        
        # Get performance data for all stocks
        performance = {}
        for stock, data in self.market_data.items():
            if stock in self.stocks:  # Only consider individual stocks, not indices
                performance[stock] = data['change']
        
        if not performance:
            logger.warning("Not enough data to identify outliers")
            return outliers
        
        # Calculate mean and standard deviation
        mean_perf = sum(performance.values()) / len(performance)
        std_perf = (sum((x - mean_perf) ** 2 for x in performance.values()) / len(performance)) ** 0.5
        
        # Identify performance outliers (beyond 1.5 standard deviations)
        for stock, change in performance.items():
            if change > mean_perf + 1.5 * std_perf:
                outliers['positive'].append({
                    'stock': stock,
                    'change': change,
                    'z_score': (change - mean_perf) / std_perf
                })
            elif change < mean_perf - 1.5 * std_perf:
                outliers['negative'].append({
                    'stock': stock,
                    'change': change,
                    'z_score': (change - mean_perf) / std_perf
                })
        
        # Sort outliers by magnitude
        outliers['positive'] = sorted(outliers['positive'], key=lambda x: x['change'], reverse=True)
        outliers['negative'] = sorted(outliers['negative'], key=lambda x: x['change'])
        
        # Identify volume outliers
        volume_data = {}
        for stock, data in self.market_data.items():
            if stock in self.stocks and 'volume' in data:
                # Compare to average volume
                hist_data = self.historical_prices.get(stock)
                if hist_data is not None and not hist_data.empty:
                    avg_volume = hist_data['Volume'].mean()
                    if avg_volume > 0:
                        volume_ratio = data['volume'] / avg_volume
                        volume_data[stock] = volume_ratio
        
        # Find stocks with unusually high volume
        for stock, ratio in volume_data.items():
            if ratio > 2.0:  # Volume more than double the average
                outliers['volume'].append({
                    'stock': stock,
                    'volume_ratio': ratio,
                    'current_volume': self.market_data[stock]['volume']
                })
        
        # Sort volume outliers
        outliers['volume'] = sorted(outliers['volume'], key=lambda x: x['volume_ratio'], reverse=True)
        
        self.analysis_results['outliers'] = outliers
        logger.info("Outlier identification completed")
        
        return outliers
    
    def generate_forward_outlook(self):
        """Generate forward-looking insights based on all analyses"""
        logger.info("Generating forward outlook...")
        
        outlook = {
            'catalysts': [],
            'trends': [],
            'vulnerabilities': [],
            'opportunities': []
        }
        
        # Identify upcoming catalysts from news
        if self.news_data:
            # Look for forward-looking keywords in news
            catalyst_keywords = ['upcoming', 'scheduled', 'expected', 'anticipate', 'forecast', 'predict', 'future', 'tomorrow', 'next week']
            
            for item in self.news_data:
                content = item['title'] + ' ' + item['content']
                if any(keyword in content.lower() for keyword in catalyst_keywords):
                    outlook['catalysts'].append({
                        'title': item['title'],
                        'source': item['source'],
                        'date': item['date']
                    })
        
        # Identify market trends based on technical analysis
        trend_count = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        sector_trends = {}
        
        for stock, analysis in self.analysis_results.get('technical', {}).items():
            signals = analysis.get('signal', [])
            bullish_signals = sum(1 for signal in signals if 'Bullish' in signal or 'Above' in signal or 'Oversold' in signal)
            bearish_signals = sum(1 for signal in signals if 'Bearish' in signal or 'Below' in signal or 'Overbought' in signal)
            
            if bullish_signals > bearish_signals:
                trend_count['bullish'] += 1
                trend = 'bullish'
            elif bearish_signals > bullish_signals:
                trend_count['bearish'] += 1
                trend = 'bearish'
            else:
                trend_count['neutral'] += 1
                trend = 'neutral'
            
            # Group by sector
            for sector, stocks in self.sectors.items():
                if stock in stocks:
                    if sector not in sector_trends:
                        sector_trends[sector] = {'bullish': 0, 'bearish': 0, 'neutral': 0}
                    sector_trends[sector][trend] += 1
        
        # Determine overall market trend
        overall_trend = max(trend_count, key=trend_count.get)
        outlook['trends'].append({
            'type': 'Overall Market',
            'trend': overall_trend,
            'strength': trend_count[overall_trend] / sum(trend_count.values()) if sum(trend_count.values()) > 0 else 0
        })
        
        # Determine sector trends
        for sector, trends in sector_trends.items():
            dominant_trend = max(trends, key=trends.get)
            trend_strength = trends[dominant_trend] / sum(trends.values()) if sum(trends.values()) > 0 else 0
            
            if trend_strength >= 0.6:  # At least 60% of stocks in sector show the same trend
                outlook['trends'].append({
                    'type': 'Sector',
                    'sector': sector,
                    'trend': dominant_trend,
                    'strength': trend_strength
                })
        
        # Identify market vulnerabilities
        # 1. Check for overbought conditions in a bullish market
        if overall_trend == 'bullish':
            overbought_stocks = []
            for stock, analysis in self.analysis_results.get('technical', {}).items():
                if analysis.get('rsi', 0) > 70:
                    overbought_stocks.append(stock)
            
            if len(overbought_stocks) > len(self.analysis_results.get('technical', {})) / 3:
                outlook['vulnerabilities'].append({
                    'type': 'Overbought Market',
                    'description': 'A significant number of stocks show overbought conditions, indicating potential for a pullback',
                    'affected_stocks': overbought_stocks[:5]  # List up to 5 examples
                })
        
        # 2. Check for divergence between market performance and sentiment
        market_performance = self.market_data.get('^NSEI', {}).get('change', 0)
        sentiment_score = self.analysis_results.get('sentiment', {}).get('overall', 0)
        
        if market_performance > 1 and sentiment_score < -0.2:
            outlook['vulnerabilities'].append({
                'type': 'Bullish Performance with Bearish Sentiment',
                'description': 'Market is rising despite negative sentiment, which may indicate unsustainable momentum'
            })
        elif market_performance < -1 and sentiment_score > 0.2:
            outlook['vulnerabilities'].append({
                'type': 'Bearish Performance with Bullish Sentiment',
                'description': 'Market is falling despite positive sentiment, which may indicate underlying structural issues'
            })
        
        # Identify potential opportunities
        # 1. Oversold stocks in generally bullish sectors
        for sector, trends in sector_trends.items():
            if trends.get('bullish', 0) > trends.get('bearish', 0):
                for stock in self.sectors[sector]:
                    analysis = self.analysis_results.get('technical', {}).get(stock, {})
                    if analysis.get('rsi', 50) < 30:
                        outlook['opportunities'].append({
                            'type': 'Oversold Stock in Bullish Sector',
                            'stock': stock,
                            'sector': sector,
                            'rsi': analysis.get('rsi')
                        })
        
        # 2. Stocks with positive sentiment but bearish technical signals
        for stock, sentiment in self.analysis_results.get('sentiment', {}).get('by_stock', {}).items():
            if sentiment > 0.3:  # Strong positive sentiment
                analysis = self.analysis_results.get('technical', {}).get(stock, {})
                signals = analysis.get('signal', [])
                bearish_signals = sum(1 for signal in signals if 'Bearish' in signal or 'Below' in signal)
                
                if bearish_signals >= 2:
                    outlook['opportunities'].append({
                        'type': 'Sentiment-Technical Divergence',
                        'stock': stock,
                        'sentiment': sentiment,
                        'bearish_signals': bearish_signals
                    })
        
        self.analysis_results['outlook'] = outlook
        logger.info("Forward outlook generated")
        
        return outlook
    
    def assess_data_confidence(self):
        """Assess confidence in the data and analysis"""
        logger.info("Assessing data confidence...")
        
        confidence = {
            'overall': 0,
            'by_source': {},
            'limitations': [],
            'recommendations': []
        }
        
        # Assess market data completeness
        market_data_count = len(self.market_data)
        expected_data_count = len(self.stocks) + len(self.indices)
        market_data_completeness = market_data_count / expected_data_count if expected_data_count > 0 else 0
        
        confidence['by_source']['market_data'] = market_data_completeness
        
        if market_data_completeness < 0.8:
            confidence['limitations'].append({
                'source': 'Market Data',
                'issue': f'Only {market_data_completeness:.0%} of expected market data was retrieved',
                'impact': 'Incomplete market analysis'
            })
            confidence['recommendations'].append('Verify connectivity to market data providers')
        
        # Assess news data freshness and volume
        news_confidence = 0
        if self.news_data:
            # Check recency - at least some news from today
            today = datetime.now().strftime('%Y-%m-%d')
            recent_news = sum(1 for item in self.news_data if today in item.get('date', ''))
            
            news_confidence = min(1.0, len(self.news_data) / 20) * 0.5  # Volume factor
            news_confidence += min(1.0, recent_news / 5) * 0.5  # Recency factor
            
            confidence['by_source']['news_data'] = news_confidence
            
            if recent_news == 0:
                confidence['limitations'].append({
                    'source': 'News Data',
                    'issue': 'No news articles from today',
                    'impact': 'Sentiment analysis may not reflect current events'
                })
                confidence['recommendations'].append('Check news API connectivity')
        else:
            confidence['by_source']['news_data'] = 0
            confidence['limitations'].append({
                'source': 'News Data',
                'issue': 'No news data retrieved',
                'impact': 'Sentiment analysis incomplete'
            })
            confidence['recommendations'].append('Verify news API credentials')
        
        # Assess social media data
        social_confidence = 0
        if self.social_media_data:
            social_confidence = min(1.0, len(self.social_media_data) / 100)
            confidence['by_source']['social_data'] = social_confidence
            
            if social_confidence < 0.5:
                confidence['limitations'].append({
                    'source': 'Social Media Data',
                    'issue': 'Limited social media data retrieved',
                    'impact': 'Social sentiment analysis may be unreliable'
                })
        else:
            confidence['by_source']['social_data'] = 0
            confidence['limitations'].append({
                'source': 'Social Media Data',
                'issue': 'No social media data retrieved',
                'impact': 'Social sentiment analysis incomplete'
            })
            confidence['recommendations'].append('Verify social media API credentials')
        
        # Assess economic indicators
        if self.economic_indicators:
            # In a real system, you'd check the last update date
            confidence['by_source']['economic_data'] = 1.0
        else:
            confidence['by_source']['economic_data'] = 0
            confidence['limitations'].append({
                'source': 'Economic Indicators',
                'issue': 'No economic data retrieved',
                'impact': 'Macroeconomic context missing'
            })
        
        # Calculate overall confidence
        source_weights = {
            'market_data': 0.4,
            'news_data': 0.3,
            'social_data': 0.2,
            'economic_data': 0.1
        }
        
        overall_confidence = sum(
            confidence['by_source'].get(source, 0) * weight
            for source, weight in source_weights.items()
        )
        
        confidence['overall'] = overall_confidence
        
        # Add specific advice based on confidence level
        if overall_confidence < 0.6:
            confidence['recommendations'].append('Consider deferring major decisions until data quality improves')
        
        self.analysis_results['confidence'] = confidence
        logger.info(f"Data confidence assessment completed: {overall_confidence:.0%}")
        
        return confidence
    
    def generate_report(self):
        """Generate a comprehensive market analysis report"""
        logger.info("Generating market report...")
        
        # Ensure we have all necessary analyses
        self.fetch_market_data()
        self.fetch_news_data()
        self.fetch_social_media_data()
        self.fetch_economic_indicators()
        self.perform_technical_analysis()
        self.analyze_sentiment()
        self.perform_correlation_analysis()
        self.identify_outliers()
        self.generate_forward_outlook()
        self.assess_data_confidence()
        
        # Create the market pulse snapshot
        market_pulse = self._generate_market_pulse()
        
        # Generate detailed analysis section
        detailed_analysis = self._generate_detailed_analysis()
        
        # Generate sentiment indicators
        sentiment_indicators = self._generate_sentiment_indicators()
        
        # Generate forward outlook
        forward_outlook = self._generate_forward_outlook()
        
        # Generate data confidence
        data_confidence = self._generate_data_confidence()
        
        # Combine into full report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_pulse': market_pulse,
            'detailed_analysis': detailed_analysis,
            'sentiment_indicators': sentiment_indicators,
            'forward_outlook': forward_outlook,
            'data_confidence': data_confidence
        }
        
        # Save report to file
        report_filename = f"data/market_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Market report generated and saved to {report_filename}")
        
        return report
    
    def _generate_market_pulse(self):
        """Generate the market pulse summary"""
        nifty = self.market_data.get('^NSEI', {})
        sensex = self.market_data.get('^BSESN', {})
        
        market_state = "bullish" if nifty.get('change', 0) > 0 else "bearish"
        sentiment = self.analysis_results.get('sentiment', {}).get('overall', 0)
        sentiment_state = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
        
        # Create the pulse message
        pulse = (
            f"Indian markets are {market_state} with NIFTY 50 at {nifty.get('current', 'N/A'):,.2f} "
            f"({nifty.get('change', 0):+.2f}%) and SENSEX at {sensex.get('current', 'N/A'):,.2f} "
            f"({sensex.get('change', 0):+.2f}%). Market sentiment is {sentiment_state}."
        )
        
        return pulse
    
    def _generate_detailed_analysis(self):
        """Generate the detailed analysis section"""
        # Get sector performance
        sector_performance = {}
        for sector, stocks in self.sectors.items():
            sector_changes = []
            for stock in stocks:
                if stock in self.market_data:
                    sector_changes.append(self.market_data[stock].get('change', 0))
            
            if sector_changes:
                sector_performance[sector] = {
                    'average_change': sum(sector_changes) / len(sector_changes),
                    'best_performer': max(sector_changes),
                    'worst_performer': min(sector_changes)
                }
        
        # Get correlation insights
        correlation_insights = []
        high_corr = self.analysis_results.get('correlation', {}).get('high_correlation', [])
        inverse_corr = self.analysis_results.get('correlation', {}).get('inverse_correlation', [])
        
        for stock1, stock2, corr in high_corr[:3]:  # Top 3 highly correlated pairs
            stock1_name = stock1.split('.')[0]
            stock2_name = stock2.split('.')[0]
            correlation_insights.append(f"{stock1_name} and {stock2_name} are highly correlated ({corr:.2f})")
        
        for stock1, stock2, corr in inverse_corr[:3]:  # Top 3 inversely correlated pairs
            stock1_name = stock1.split('.')[0]
            stock2_name = stock2.split('.')[0]
            correlation_insights.append(f"{stock1_name} and {stock2_name} are inversely correlated ({corr:.2f})")
        
        # Get outlier insights
        outlier_insights = []
        positive_outliers = self.analysis_results.get('outliers', {}).get('positive', [])
        negative_outliers = self.analysis_results.get('outliers', {}).get('negative', [])
        volume_outliers = self.analysis_results.get('outliers', {}).get('volume', [])
        
        for outlier in positive_outliers[:3]:
            stock_name = outlier['stock'].split('.')[0]
            outlier_insights.append(f"{stock_name} is outperforming ({outlier['change']:+.2f}%)")
        
        for outlier in negative_outliers[:3]:
            stock_name = outlier['stock'].split('.')[0]
            outlier_insights.append(f"{stock_name} is underperforming ({outlier['change']:+.2f}%)")
        
        for outlier in volume_outliers[:3]:
            stock_name = outlier['stock'].split('.')[0]
            outlier_insights.append(f"{stock_name} has unusual volume ({outlier['volume_ratio']:.1f}x average)")
        
        # Compile the detailed analysis
        detailed_analysis = {
            'sector_performance': sector_performance,
            'correlation_insights': correlation_insights,
            'outlier_insights': outlier_insights,
            'economic_context': self.economic_indicators
        }
        
        return detailed_analysis
    
    def _generate_sentiment_indicators(self):
        """Generate the sentiment indicators section"""
        sentiment_data = self.analysis_results.get('sentiment', {})
        
        # Determine overall sentiment rating
        overall_sentiment = sentiment_data.get('overall', 0)
        if overall_sentiment > 0.2:
            sentiment_rating = "Bullish"
        elif overall_sentiment < -0.2:
            sentiment_rating = "Bearish"
        else:
            sentiment_rating = "Neutral"
        
        # Find sentiment outliers (sectors with notably different sentiment)
        sentiment_outliers = []
        for sector, score in sentiment_data.get('by_sector', {}).items():
            if abs(score - overall_sentiment) > 0.3:
                direction = "more positive" if score > overall_sentiment else "more negative"
                sentiment_outliers.append(f"{sector} sentiment is {direction} than market average")
        
        # Extract key narratives
        narratives = sentiment_data.get('narratives', [])
        
        # Compile sentiment indicators
        sentiment_indicators = {
            'overall_rating': sentiment_rating,
            'news_sentiment': sentiment_data.get('news', 0),
            'social_sentiment': sentiment_data.get('social', 0),
            'sentiment_outliers': sentiment_outliers,
            'key_narratives': narratives
        }
        
        return sentiment_indicators
    
    def _generate_forward_outlook(self):
        """Generate the forward outlook section"""
        outlook = self.analysis_results.get('outlook', {})
        
        # Extract the most important elements for the report
        watch_items = []
        for catalyst in outlook.get('catalysts', [])[:3]:
            watch_items.append(f"Upcoming: {catalyst['title']}")
        
        emerging_patterns = []
        for trend in outlook.get('trends', []):
            if trend['type'] == 'Sector':
                strength = "strong" if trend['strength'] > 0.8 else "moderate"
                emerging_patterns.append(f"{trend['sector']} showing {strength} {trend['trend']} trend")
            elif trend['type'] == 'Overall Market':
                emerging_patterns.append(f"Overall market trend: {trend['trend']}")
        
        risk_factors = []
        for vulnerability in outlook.get('vulnerabilities', []):
            risk_factors.append(vulnerability['description'])
        
        opportunities = []
        for opportunity in outlook.get('opportunities', [])[:3]:
            if opportunity['type'] == 'Oversold Stock in Bullish Sector':
                stock_name = opportunity['stock'].split('.')[0]
                opportunities.append(f"{stock_name} potentially oversold (RSI: {opportunity['rsi']:.1f}) in bullish {opportunity['sector']} sector")
            elif opportunity['type'] == 'Sentiment-Technical Divergence':
                stock_name = opportunity['stock'].split('.')[0]
                opportunities.append(f"{stock_name} has positive sentiment despite bearish technical signals")
        
        # Compile forward outlook
        forward_outlook = {
            'watch_items': watch_items,
            'emerging_patterns': emerging_patterns,
            'risk_factors': risk_factors,
            'opportunities': opportunities
        }
        
        return forward_outlook
    
    def _generate_data_confidence(self):
        """Generate the data confidence section"""
        confidence = self.analysis_results.get('confidence', {})
        
        # Format the confidence metrics
        data_confidence = {
            'overall_confidence': confidence.get('overall', 0),
            'source_reliability': {
                'Market Data': confidence.get('by_source', {}).get('market_data', 0),
                'News Data': confidence.get('by_source', {}).get('news_data', 0),
                'Social Media': confidence.get('by_source', {}).get('social_data', 0),
                'Economic Indicators': confidence.get('by_source', {}).get('economic_data', 0)
            },
            'limitations': [item['issue'] for item in confidence.get('limitations', [])],
            'recommendations': confidence.get('recommendations', [])
        }
        
        return data_confidence
    
    def format_response(self, report):
        """Format the analysis report according to the specified format"""
        market_pulse = report['market_pulse']
        detailed = report['detailed_analysis']
        sentiment = report['sentiment_indicators']
        outlook = report['forward_outlook']
        confidence = report['data_confidence']
        
        response = f"[MARKET PULSE]: {market_pulse}\n\n"
        
        response += "[DETAILED ANALYSIS]\n"
        for sector, perf in detailed['sector_performance'].items():
            response += f"- {sector}: {perf['average_change']:+.2f}% (Range: {perf['worst_performer']:+.2f}% to {perf['best_performer']:+.2f}%)\n"
        
        for insight in detailed['correlation_insights']:
            response += f"- {insight}\n"
            
        for insight in detailed['outlier_insights']:
            response += f"- {insight}\n"
            
        response += f"- INR/USD: {detailed['economic_context'].get('INR_USD', 'N/A')}, Repo Rate: {detailed['economic_context'].get('Repo_Rate', 'N/A')}%\n\n"
        
        response += "[SENTIMENT INDICATORS]\n"
        response += f"- Overall sentiment: {sentiment['overall_rating']} (News: {sentiment['news_sentiment']:.2f}, Social: {sentiment['social_sentiment']:.2f})\n"
        
        for outlier in sentiment['sentiment_outliers']:
            response += f"- {outlier}\n"
            
        for narrative in sentiment['key_narratives'][:3]:
            response += f"- {narrative['title']} ({narrative['source']}, Sentiment: {narrative['sentiment']:.2f})\n\n"
        
        response += "[FORWARD OUTLOOK]\n"
        for item in outlook['watch_items']:
            response += f"- {item}\n"
            
        for pattern in outlook['emerging_patterns']:
            response += f"- {pattern}\n"
            
        for risk in outlook['risk_factors']:
            response += f"- Risk: {risk}\n"
            
        for opportunity in outlook['opportunities']:
            response += f"- Opportunity: {opportunity}\n\n"
        
        response += "[DATA CONFIDENCE]\n"
        response += f"- Overall confidence: {confidence['overall_confidence']:.0%}\n"
        
        for source, reliability in confidence['source_reliability'].items():
            response += f"- {source} reliability: {reliability:.0%}\n"
            
        if confidence['limitations']:
            response += f"- Key limitations: {confidence['limitations'][0]}\n"
            
        for recommendation in confidence['recommendations'][:2]:
            response += f"- {recommendation}\n"
        
        return response
    
    def run_continuous_analysis(self, interval_minutes=15):
        """Run continuous analysis at specified intervals"""
        def scheduled_task():
            logger.info(f"Running scheduled market analysis at {datetime.now()}")
            report = self.generate_report()
            response = self.format_response(report)
            
            # In a real application, you might send this to subscribers
            # or store it for retrieval via an API
            print("\n" + "="*80 + "\n")
            print(response)
            print("\n" + "="*80 + "\n")
            
            logger.info("Scheduled analysis completed")
        
        # Run immediately first
        scheduled_task()
        
        # Then schedule for regular intervals
        schedule.every(interval_minutes).minutes.do(scheduled_task)
        
        # Keep running in a loop
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for pending tasks
    
    def run_single_analysis(self):
        """Run a single analysis cycle and return formatted results"""
        report = self.generate_report()
        return self.format_response(report)


if __name__ == "__main__":
    try:
        # Create the agent
        agent = IndianFinancialMarketAgent()
        
        # Choose whether to run continuous or single analysis
        continuous_mode = False
        
        if continuous_mode:
            # Run continuous analysis (15-minute intervals)
            agent.run_continuous_analysis(interval_minutes=15)
        else:
            # Run a single analysis and print results
            results = agent.run_single_analysis()
            print(results)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)