
import random
import time

class OilTradingDesk:
    """
    Oil Refinement Trading Module
    Simulates spot prices, futures contracts, and order books
    """
    def __init__(self):
        self.spot_price = 78.50  # USD/barrel
        self.order_book = []
        self.history = []
        
    def generate_market_data(self):
        """Update market simulation"""
        change = random.uniform(-1.5, 1.8)
        self.spot_price += change
        self.spot_price = max(40.0, self.spot_price)
        
        timestamp = time.strftime('%H:%M:%S')
        self.history.append({'time': timestamp, 'price': self.spot_price})
        if len(self.history) > 50:
            self.history.pop(0)
            
        return {
            'spot_price_usd': round(self.spot_price, 2),
            'brent_crude': round(self.spot_price + 2.5, 2),
            'wti_crude': round(self.spot_price - 1.2, 2),
            'trend': 'Bullish' if change > 0 else 'Bearish',
            'change_percent': round((change/self.spot_price)*100, 2)
        }
        
    def place_order(self, type, volume, price_limit=None):
        order = {
            'id': random.randint(1000, 9999),
            'type': type, # BUY/SELL
            'volume_mbbl': volume, # Thousand barrels
            'status': 'Executed',
            'price': self.spot_price if not price_limit else price_limit
        }
        self.order_book.append(order)
        return order

    def get_order_book(self):
        return self.order_book[-10:] # Last 10 orders
