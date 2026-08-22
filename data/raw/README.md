# data/raw — official competition files go here (never committed)

Expected contents (verify against the official documentation):

- training time-series (Parquet): `timestamp, battery_id, voltage, temperature`
- evaluation time-series (same schema, truncated, EOL hidden)
- locations table: `building_id, room_id, battery_id`
- travel-time matrix: building × building, minutes

`make data` validates whatever is placed here against the schemas in
`src/batteryswap/io.py`.
