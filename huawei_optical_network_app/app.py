"""
Huawei Optical Networking App
Simulates and configures next-gen optical networks with quantum/physics-based models.
"""
from optical_switch import OpticalSwitch
from network_simulation import OpticalNetworkSimulator
from router_config import QuantumRouterConfig

def main():
    # Example usage
    switch = OpticalSwitch(1.5, 1.33)
    print("Critical angle:", switch.critical_angle())
    print(switch.simulate_path_integral(1550, 10))
    print(switch.continued_fraction_loss())

    sim = OpticalNetworkSimulator([switch], 800)
    print("Network simulation:", sim.simulate())

    router = QuantumRouterConfig("mesh", {"entanglement": True})
    print(router.configure())

if __name__ == "__main__":
    main()