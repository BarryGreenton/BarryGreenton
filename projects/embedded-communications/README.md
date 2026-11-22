# Embedded communications on a vehicle controller

How does an embedded controller exchange state with the devices attached to it?

This engineering placement concerned the serial communication layer of a shared vehicle controller. The project materials include an RS-485 controller protocol, a Modbus battery-management specification and a C firmware tree with separate device and communication modules.

This public case study describes the interfaces. Hardware integration uses a host, a controller and a battery-management endpoint. The employer’s firmware, device-specific frames and protocol manuals stay in the original archive.

### System boundaries

| Component | Function in the project |
| --- | --- |
| Host | Sends a request and displays the controller’s response for inspection. |
| Controller firmware | Handles the incoming serial message, dispatches it to the corresponding module and returns the response. |
| Device interface | Carries controller or battery-management information across the bus. |

The distinction between the RS-485 electrical interface and Modbus message handling matters during debugging. A visible response establishes communication. Its contents still have to agree with the requested operation.

[Return to the portfolio](../../README.md)
