# Module Parameters
| Module | Parameter | Value |
| --- | --- | --- |
| `accel\|LIS3DHTR_SensorBlock.c_bulk\|Capacitor` | `capacitance` | `([9µF, 11µF])` |
|  | `max_voltage` | `([10V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X5R]` |
| `accel\|LIS3DHTR_SensorBlock.c_hf\|Capacitor` | `capacitance` | `([90nF, 110nF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X7R]` |
| `c_nrst\|Capacitor` | `capacitance` | `([90nF, 110nF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X7R]` |
| `humidity\|HDC2080_SensorBlock.c_bypass\|Capacitor` | `capacitance` | `([90nF, 110nF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X7R]` |
| `i2c\|I2C_BusBlock.r_scl\|Resistor` | `max_power` | `([62.5mW])` |
|  | `max_voltage` | `([50V])` |
|  | `resistance` | `([4.653kΩ, 4.747kΩ])` |
| `i2c\|I2C_BusBlock.r_sda\|Resistor` | `max_power` | `([62.5mW])` |
|  | `max_voltage` | `([50V])` |
|  | `resistance` | `([4.653kΩ, 4.747kΩ])` |
| `imu\|QMI8658C_IMUBlock.cp1\|Capacitor` | `capacitance` | `([90nF, 110nF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X7R]` |
| `imu\|QMI8658C_IMUBlock.cp2\|Capacitor` | `capacitance` | `([90nF, 110nF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X7R]` |
| `l1\|Inductor` | `dc_resistance` | `([1.17Ω])` |
|  | `inductance` | `([8µH, 12µH])` |
|  | `max_current` | `([50mA])` |
|  | `saturation_current` | `([50mA])` |
|  | `self_resonant_frequency` | `([30MHz])` |
| `led\|StatusLED_Block.r_limit\|Resistor` | `max_power` | `([62.5mW])` |
|  | `max_voltage` | `([50V])` |
|  | `resistance` | `([990Ω, 1.01kΩ])` |
| `ntc\|NTC_TempSensor.r_fixed\|Resistor` | `max_power` | `([62.5mW])` |
|  | `max_voltage` | `([50V])` |
|  | `resistance` | `([9.9kΩ, 10.1kΩ])` |
| `power\|PowerFilterBlock.c_bulk\|Capacitor` | `capacitance` | `([17.6µF, 26.4µF])` |
|  | `max_voltage` | `([25V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X5R]` |
| `power\|PowerFilterBlock.c_mcu_1\|Capacitor` | `capacitance` | `([90nF, 110nF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X7R]` |
| `power\|PowerFilterBlock.c_mcu_2\|Capacitor` | `capacitance` | `([4.23µF, 5.17µF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X5R]` |
| `power\|PowerFilterBlock.c_sensor_1\|Capacitor` | `capacitance` | `([90nF, 110nF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X7R]` |
| `power\|PowerFilterBlock.c_sensor_2\|Capacitor` | `capacitance` | `([90nF, 110nF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X7R]` |
| `power\|PowerFilterBlock.c_vcc_1\|Capacitor` | `capacitance` | `([4.23µF, 5.17µF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X5R]` |
| `power\|PowerFilterBlock.c_vcc_2\|Capacitor` | `capacitance` | `([90nF, 110nF])` |
|  | `max_voltage` | `([16V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.X7R]` |
| `r_activation\|Resistor` | `max_power` | `([62.5mW])` |
|  | `max_voltage` | `([50V])` |
|  | `resistance` | `([9.9kΩ, 10.1kΩ])` |
| `rf\|RF_MatchingNetwork.c_harmonic\|Capacitor` | `capacitance` | `([3.234pF, 3.366pF])` |
|  | `max_voltage` | `([50V])` |
|  | `temperature_coefficient` | `[TemperatureCoefficient.C0G]` |
| `rf\|RF_MatchingNetwork.l_harmonic\|Inductor` | `dc_resistance` | `([400mΩ])` |
|  | `inductance` | `([7.79nH, 8.61nH])` |
|  | `max_current` | `([300mA])` |
|  | `saturation_current` | `([0A, InfinityA])` |
|  | `self_resonant_frequency` | `([3.6GHz])` |
| `rf\|RF_MatchingNetwork.l_series\|Inductor` | `dc_resistance` | `([44mΩ])` |
|  | `inductance` | `([4.116nH, 4.284nH])` |
|  | `max_current` | `([1.8A])` |
|  | `saturation_current` | `([0A, InfinityA])` |
|  | `self_resonant_frequency` | `([0Hz, InfinityHz])` |

