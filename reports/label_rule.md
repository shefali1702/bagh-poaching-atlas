# Proxy label rule

Exact poaching coordinates in India are confidential, so every 1 km cell gets a **proxy threat score**
built from a documented rule. This is the proxy-label approach used in conservation ML when ground truth is restricted.

```
latent = 1.30 * exp(-dist_to_boundary / 3 km)          # reserve edges are porous
       + 1.10 * exp(-dist_to_road / 6 km)              # vehicle access
       + 0.70 * exp(-dist_to_town / 20 km)             # markets and poacher bases
       + 0.90 * log-scaled target-species sightings    # where the animals are (GBIF)
       + 0.90 * dry * exp(-dist_to_water / 2.5 km)     # dry-season waterhole concentration
       + 0.20 * wet * exp(-dist_to_water / 2.5 km)
       + 0.60 * exp(-dist_to_border / 60 km)           # trafficking routes
       + 0.80 * fragmentation_index
       + 1.20 * reserve_pressure                       # past poaching pressure of the reserve
       + 0.55 * (1 - exp(-dist_to_ranger / 5 km))      # far from patrol posts
       - 0.012 * slope_deg                             # steep ground slows poachers too
       + Normal(0, 0.28)                               # noise, so the rule cannot be recovered exactly

threat_score = 100 / (1 + exp(-1.6 * (z(latent) - 0.35)))      # 0-100
risk_label   = High (top 15%), Medium (next 30%), Low (rest)
```

## Why these weights

The order of the weights follows the drivers that published poaching studies keep finding: access (edges, roads,
towns), where the target animals are, water in the dry season, governance and pressure, and patrol coverage.
The exact numbers are a judgement call. Say so in the viva and show the SHAP plot: the model recovers this
order without being told it.

## Replacing the proxy with real data

1. Run `python -m src.ntca_scraper` on your own machine to get tiger deaths per reserve from NTCA.
2. Replace the simulated `reserve_pressure` with the real poaching rate per reserve.
3. Use CITES MIKE (elephants) and Assam Forest Department counts (rhinos) the same way.
4. Check that reserves with a higher real poaching rate also get more predicted high-risk cells.
