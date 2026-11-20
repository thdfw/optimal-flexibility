def fahrenheit_to_kelvin(temp_f: float) -> float:
    return (temp_f - 32) * 5/9 + 273.15

def kelvin_to_fahrenheit(temp_kelvin: float) -> float:
    return (temp_kelvin - 273.15) * 9/5 + 32