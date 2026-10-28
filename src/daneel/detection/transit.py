model = batman.TransitModel(params, t)  # initialise the transit model
flux = model.light_curve(params)           # calculate the normalised flux

assert flux.shape == t.shape
assert np.all(np.isfinite(flux)), "The light curve contains non-finite values"
print(f"Minimum model flux: {flux.min():.6f}")
print(f"Maximum model flux: {flux.max():.6f}")
