"""
Interactive Brokers (IBKR) Integration Client
---------------------------------------------
Simulates connection to IB Gateway or TWS via IB API (ibapi).
Provides execution capability for the optimized portfolio.
"""

import time
import logging
from typing import Dict, List, Optional

# Mocking ibapi for simulation if not installed
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    IBAPI_INSTALLED = True
except ImportError:
    IBAPI_INSTALLED = False
    # Define dummy classes for type hinting/structure
    class EClient: pass
    class EWrapper: pass
    class Contract: pass
    class Order: pass

class IBKRWrapper(EWrapper):
    """Callback handler for IBKR events"""
    def __init__(self):
        super().__init__()
        self.next_order_id = None

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_order_id = orderId
        print(f"IBKR: Next Valid Order ID: {orderId}")

class IBKRClient:
    """
    Client for interacting with Interactive Brokers.
    Handles order placement, market data requests, and account summary.
    """
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.simulation_mode = not IBAPI_INSTALLED
        self.connected = False
        
        if not self.simulation_mode:
            self.wrapper = IBKRWrapper()
            self.client = EClient(self.wrapper)
        
    def connect(self):
        """Connect to IB TWS or Gateway"""
        if self.simulation_mode:
            print("IBKR: Running in SIMULATION mode (ibapi not installed or TWS offline)")
            self.connected = True
            return True
            
        try:
            self.client.connect(self.host, self.port, self.client_id)
            # Start client thread in background in real impl
            print("IBKR: Connected to TWS")
            self.connected = True
            return True
        except Exception as e:
            print(f"IBKR: Connection failed: {e}")
            self.simulation_mode = True
            return False

    def get_market_data(self, symbol: str, sec_type: str = 'STK', currency: str = 'USD') -> Dict:
        """Get real-time market data snapshot"""
        if self.simulation_mode:
            # Return mock data based on our local generator
            return {
                 'symbol': symbol,
                 'bid': 150.05,
                 'ask': 150.10,
                 'last': 150.07,
                 'volume': 1500000
            }
        # Real impl would reqMktData
        return {}

    def execute_portfolio_rebalance(self, holdings: List[Dict]) -> Dict:
        """
        Execute a batch of orders to rebalance portfolio.
        Algo: 'Adaptive' or 'VWAP' to minimize market impact.
        """
        orders_placed = []
        total_value = 0
        
        print(f"IBKR: Analyzing {len(holdings)} holdings for potential execution...")
        
        for holding in holdings:
            symbol = holding['symbol']
            action = 'BUY' if holding['weight'] > 0 else 'SELL'
            quantity = holding['shares']
            
            if quantity == 0: continue
            
            # Create Contract
            contract = self._create_contract(symbol)
            
            # Create Order (Algo: Adaptive)
            order = self._create_order(action, quantity, algo_strategy='Adaptive')
            
            # Determine execution venue/exchange
            exchange = 'SMART' # IB SmartRouting associated with best execution
            if symbol.endswith('.TO'): exchange = 'TSE'
            
            # Place Order
            order_id = int(time.time() * 1000) % 1000000 # Mock ID
            print(f"IBKR: Placing {action} {quantity} {symbol} @ {exchange} (Algo: Adaptive)")
            
            orders_placed.append({
                'order_id': order_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'status': 'Submitted',
                'exchange': exchange
            })
            total_value += holding['value']
            
        return {
            'status': 'submitted',
            'orders': orders_placed,
            'broker': 'Interactive Brokers',
            'execution_start': time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def _create_contract(self, symbol: str) -> Contract:
        """Helper to create IB Contract object"""
        if self.simulation_mode: return None
        
        c = Contract()
        c.symbol = symbol.replace('.TO', '') # IB doesn't use extension for symbol usually
        c.secType = 'STK'
        c.currency = 'CAD' if symbol.endswith('.TO') else 'USD'
        c.exchange = 'SMART'
        return c

    def _create_order(self, action: str, quantity: float, algo_strategy: str = None) -> Order:
        """Helper to create IB Order object"""
        if self.simulation_mode: return None
        
        o = Order()
        o.action = action
        o.totalQuantity = quantity
        o.orderType = 'MKT'
        
        if algo_strategy == 'Adaptive':
            o.algoStrategy = 'Adaptive'
            o.algoParams = [] # Add params
            
        return o
