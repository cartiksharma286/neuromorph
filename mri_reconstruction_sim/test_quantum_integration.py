#!/usr/bin/env python3
"""
Test script to verify quantum noise reduction integration with Flask app
"""
import sys
sys.path.insert(0, '.')

try:
    from app import app
    print('✅ Flask app imported successfully with quantum module')
    
    # List all routes
    quantum_routes = [rule for rule in app.url_map.iter_rules() if 'quantum' in rule.rule]
    
    if quantum_routes:
        print(f'\n✅ Found {len(quantum_routes)} quantum endpoints:')
        for rule in quantum_routes:
            methods = ', '.join([m for m in rule.methods if m not in ('HEAD', 'OPTIONS')])
            print(f'   • {rule.rule:40} [{methods}]')
        print('\n✅ All quantum routes successfully registered!')
    else:
        print('\n⚠️  No quantum endpoints found')
        
except Exception as e:
    import traceback
    print(f'❌ Error: {e}')
    traceback.print_exc()
    sys.exit(1)
