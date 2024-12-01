import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np

# Load the cleaned dataset
file_path = 'Cleaned_Full_Data_AvgGround_Air_Surface_Moisture.csv'
cleaned_data = pd.read_csv(file_path)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Extract relevant columns (depths) and parse temperatures (Avg Ground Temp)
depth_columns = [col for col in cleaned_data.columns if col != "Time"]
time_data = pd.to_datetime(cleaned_data["Time"])
depth_values = [float(col.strip("m")) for col in depth_columns]  # Convert column names to float depths

# Create a DataFrame to store Avg Ground Temp for each depth
temperature_data = pd.DataFrame({"Time": time_data})
for col in depth_columns:
    # Extract Avg Ground Temp from each depth column
    temperature_data[col] = cleaned_data[col].apply(lambda x: eval(x)[0])  # Avg Ground Temp is the first entry

# Normalize time as days since the first observation
temperature_data["Days"] = (temperature_data["Time"] - temperature_data["Time"].min()).dt.days

# Reshape data for PINN
reshaped_data = pd.melt(
    temperature_data,
    id_vars=["Days"],
    value_vars=depth_columns,
    var_name="Depth",
    value_name="Temperature"
)

# Convert depth to float values for numerical input
reshaped_data["Depth"] = reshaped_data["Depth"].str.strip("m").astype(float)
print("Reshaping other data")

# Extract other features (Air Temp, Surface Temp, Soil Moisture)
features = pd.DataFrame()
for feature_idx, feature_name in enumerate(["Air Temp", "Surface Temp", "Soil Moisture"]):
    features[feature_name] = cleaned_data[depth_columns].map(lambda x: eval(x)[feature_idx + 1]).stack()

# Reset index to align with reshaped_data
features = features.reset_index(drop=True)

# Add features directly to reshaped_data
reshaped_data = reshaped_data.reset_index(drop=True)
reshaped_data = pd.concat([reshaped_data, features], axis=1)
print("Reshape Complete")


# Define the PINN model
class PINN(nn.Module):
    def __init__(self, input_dim=5):  # Depth, Time, Air Temp, Surface Temp, Soil Moisture
        super(PINN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Temperature as output
        )

    def forward(self, x):
        return self.layers(x)



# Define the physics loss
def physics_loss(model, depth, time, air_temp, surface_temp, soil_moisture, k=1.0, c=1.0, L=1.0):
    depth = depth.clone().detach().requires_grad_().to(device)
    time = time.clone().detach().requires_grad_().to(device)

    # Prepare inputs
    input_tensor = torch.cat((depth, time, air_temp, surface_temp, soil_moisture), dim=1)
    temp_pred = model(input_tensor)

    # Compute gradients
    temp_t = grad(temp_pred, time, grad_outputs=torch.ones_like(temp_pred).to(device), create_graph=True)[0]
    temp_x = grad(temp_pred, depth, grad_outputs=torch.ones_like(temp_pred).to(device), create_graph=True)[0]
    temp_xx = grad(temp_x, depth, grad_outputs=torch.ones_like(temp_x).to(device), create_graph=True)[0]

    # Physics loss components
    time_term = c * temp_t
    spatial_term = k * temp_xx

    # Compute physical loss
    return torch.mean((time_term - spatial_term) ** 2)

def boundary_loss(model, boundary_conditions, k):
    surface_input = boundary_conditions["surface"]["input"]
    surface_value = boundary_conditions["surface"]["value"]

    bottom_input = boundary_conditions["bottom"]["input"].clone().detach().requires_grad_().to(device)
    bottom_value = boundary_conditions["bottom"]["value"]

    # Surface boundary condition
    surface_pred = model(surface_input)
    surface_loss = torch.mean((surface_pred - surface_value) ** 2)

    # Bottom boundary condition
    bottom_pred = model(bottom_input)
    bottom_grad = grad(
        outputs=bottom_pred,
        inputs=bottom_input[:, 0:1],
        grad_outputs=torch.ones_like(bottom_pred).to(device),
        create_graph=True,
        allow_unused=True
    )[0]

    if bottom_grad is None:
        bottom_grad = torch.zeros_like(bottom_input[:, 0:1])  # Default fallback for zero gradient

    bottom_loss = torch.mean((k * bottom_grad + bottom_value) ** 2)
    return surface_loss + bottom_loss


def initial_loss(model, initial_conditions):
    initial_input = initial_conditions["input"]
    initial_value = initial_conditions["value"]

    initial_pred = model(initial_input)
    return torch.mean((initial_pred - initial_value) ** 2)


# Define the total loss
def total_loss(model, depth, time, air_temp, surface_temp, soil_moisture, temp_obs, boundary_conditions, initial_conditions, lambdas, k):
    # Data loss
    input_tensor = torch.cat((depth, time, air_temp, surface_temp, soil_moisture), dim=1).to(device)
    temp_pred = model(input_tensor)
    data_loss = torch.mean((temp_pred - temp_obs) ** 2)

    # Compute individual losses
    phys_loss = physics_loss(model, depth, time, air_temp, surface_temp, soil_moisture, k=k)
    b_loss = boundary_loss(model, boundary_conditions, k=k)
    i_loss = initial_loss(model, initial_conditions)

    # Combine losses
    return (
        lambdas["data"] * data_loss +
        lambdas["physics"] * phys_loss +
        lambdas["boundary"] * b_loss +
        lambdas["initial"] * i_loss
    )


# Convert to tensors with requires_grad=True for computing derivatives
depth_tensor = torch.tensor(reshaped_data["Depth"].values, dtype=torch.float32).unsqueeze(1).to(device)
time_tensor = torch.tensor(reshaped_data["Days"].values, dtype=torch.float32).unsqueeze(1).to(device)
temperature_tensor = torch.tensor(reshaped_data["Temperature"].values, dtype=torch.float32).unsqueeze(1).to(device)
air_temp_tensor = torch.tensor(reshaped_data["Air Temp"].values, dtype=torch.float32).unsqueeze(1).to(device)
surface_temp_tensor = torch.tensor(reshaped_data["Surface Temp"].values, dtype=torch.float32).unsqueeze(1).to(device)
soil_moisture_tensor = torch.tensor(reshaped_data["Soil Moisture"].values, dtype=torch.float32).unsqueeze(1).to(device)

surface_depth = min(depth_values)  # Surface depth (e.g., 0 m)
bottom_depth = max(depth_values)  # Bottom depth

boundary_conditions = {
    "surface": {
        "input": torch.tensor(
            [[surface_depth, t, 0, 0, 0] for t in reshaped_data["Days"].unique()],
            dtype=torch.float32
        ).to(device),
        "value": torch.tensor(
            reshaped_data[reshaped_data["Depth"] == surface_depth]["Temperature"].tolist(),
            dtype=torch.float32
        ).to(device)
    },
    "bottom": {
        "input": torch.tensor(
            [[bottom_depth, t, 0, 0, 0] for t in reshaped_data["Days"].unique()],
            dtype=torch.float32
        ).to(device),
        "value": torch.tensor(
            reshaped_data[reshaped_data["Depth"] == bottom_depth]["Temperature"].tolist(),
            dtype=torch.float32
        ).to(device)
    }
}

initial_conditions = {
    "input": torch.tensor(
        [[d, reshaped_data["Days"].min(), 0, 0, 0] for d in depth_values],
        dtype=torch.float32
    ).to(device),
    "value": torch.tensor(
        reshaped_data[reshaped_data["Days"] == reshaped_data["Days"].min()]["Temperature"].tolist(),
        dtype=torch.float32
    ).to(device)
}

#%% Training TEsting
torch.manual_seed(44) # setting seed for consistency

# Initialize the model, optimizer, and hyperparameters
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lambdas = {"data": 1.2, "physics": 0.8, "boundary": 0.6, "initial": 0.4}

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = total_loss(
        model, depth_tensor, time_tensor, air_temp_tensor, surface_temp_tensor, soil_moisture_tensor,
        temperature_tensor, boundary_conditions, initial_conditions, lambdas, k=0.1
    )
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

#%% Plotting
# Function to plot temperature profile
def plot_max_predicted_temperature(model, depth_range, time_range, device, avg_features, title="Maximum Predicted Temperature"):
    # Generate depths and times for the prediction
    depth_values = np.linspace(depth_range[0], depth_range[-1], 100)  # 100 depth points
    time_values = np.linspace(time_range[0], time_range[-1], 100)    # 100 time points

    max_temperatures = []

    for depth in depth_values:
        # Repeat depth for all time points
        depth_tensor = torch.full((len(time_values), 1), depth, dtype=torch.float32).to(device)
        time_tensor = torch.tensor(time_values, dtype=torch.float32).unsqueeze(1).to(device)

        # Use average feature values
        air_temp = torch.full_like(depth_tensor, avg_features["Air Temp"]).to(device)
        surface_temp = torch.full_like(depth_tensor, avg_features["Surface Temp"]).to(device)
        soil_moisture = torch.full_like(depth_tensor, avg_features["Soil Moisture"]).to(device)

        # Create input tensor for the model
        input_tensor = torch.cat((depth_tensor, time_tensor, air_temp, surface_temp, soil_moisture), dim=1)
        predicted_temp = model(input_tensor).detach().cpu().numpy()

        # Find the maximum predicted temperature for this depth
        max_temperatures.append(np.max(predicted_temp))

    # Plot the maximum predicted temperature vs depth
    plt.figure(figsize=(8, 6))
    plt.plot(max_temperatures, depth_values, label="Max Predicted Temperature")
    plt.gca().invert_yaxis()  # Invert depth axis to have surface at the top
    plt.xlabel("Maximum Predicted Temperature (°C)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Calculate average feature values for the dataset
avg_features = reshaped_data[["Air Temp", "Surface Temp", "Soil Moisture"]].mean()

# Plotting maximum predicted temperature for the entire depth range
plot_max_predicted_temperature(
    model,
    depth_range=(min(depth_values), max(depth_values)),
    time_range=(reshaped_data["Days"].min(), reshaped_data["Days"].max()),
    device=device,
    avg_features=avg_features,
    title="Maximum Predicted Temperature Across Depths"
)

def plot_geothermal_gradient(model, depth_range, time_range, device, avg_features, title="Geothermal Gradient"):
    # Generate depths and times for the prediction
    depth_values = np.linspace(depth_range[0], 2.25, 100)  # 100 depth points
    time_values = np.linspace(time_range[0], time_range[-1], 100)    # 100 time points

    geothermal_gradients = []

    for depth in depth_values:
        # Repeat depth for all time points
        depth_tensor = torch.full((len(time_values), 1), depth, dtype=torch.float32).to(device).requires_grad_()
        time_tensor = torch.tensor(time_values, dtype=torch.float32).unsqueeze(1).to(device)

        # Use average feature values
        air_temp = torch.full_like(depth_tensor, avg_features["Air Temp"]).to(device)
        surface_temp = torch.full_like(depth_tensor, avg_features["Surface Temp"]).to(device)
        soil_moisture = torch.full_like(depth_tensor, avg_features["Soil Moisture"]).to(device)

        # Create input tensor for the model
        input_tensor = torch.cat((depth_tensor, time_tensor, air_temp, surface_temp, soil_moisture), dim=1)
        predicted_temp = model(input_tensor)

        # Compute the gradient of temperature with respect to depth
        temp_grad = grad(predicted_temp, depth_tensor, grad_outputs=torch.ones_like(predicted_temp), create_graph=True)[0]
        geothermal_gradients.append(temp_grad.mean().detach().cpu().item())

    # Plot the geothermal gradient vs depth
    plt.figure(figsize=(8, 6))
    plt.plot(geothermal_gradients, depth_values, label="Geothermal Gradient")
    plt.gca().invert_yaxis()  # Invert depth axis to have surface at the top
    plt.axvline(-0.1, color='red', linestyle='--', label="Near Zero Gradient (Stable Layer)")
    plt.xlabel("Geothermal Gradient (°C/m)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Plotting geothermal gradient for the entire depth range
plot_geothermal_gradient(
    model,
    depth_range=(min(depth_values), max(depth_values)),
    time_range=(reshaped_data["Days"].min(), reshaped_data["Days"].max()),
    device=device,
    avg_features=avg_features,
    title="Geothermal Gradient Across Depths"
)

# Saving the entire model
torch.save(model, "model_full.pth")
