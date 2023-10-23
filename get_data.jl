#=
GET RAW DATA

This is an example repository showing you how to download data from the ECMWF Climate Data Store (CDS) using the CDS API.

As described in the documentation (https://github.com/JuliaClimate/CDSAPI.jl) You will need an API key stored on a file.
See https://cds.climate.copernicus.eu/api-how-to for instructions.

We will use data from the ERA5 reanalysis project.
A reanalysis is a gridded reconstruction of historical weather data using a fixed data assimilation system.
In essence, it combines a model and observations to reconstruct the entire state of the Earth system at a given time.

All the data we will use falls into two categories:

- reanalysis-era5-single-levels: this is for data that does not vary in the vertical dimension. Surface temperature is an example.
- reanalysis-era5-pressure-levels: this is for data that varies in the vertical dimension. Geopotential height is an example.

The following script provides you with a function to download data from both categories.
We download data one year at a time to avoid creating files that are too large.

You may want to modifty some of the arguments

- time: the functions below provide hourly data. You can modify to store only a single hour per day by setting this to "2:00" for example. Daily data is not available, but you can download hourly data and average it yourself.
- area: the area of interest. The default roughly spans the continental US, but you can modify
- grid: the grid resolution. The default is 1 degree, but you can modify to increase or lower resolution
- pressure_level: only used for pressure level data. Here we use 500 hPa, but you can modify to download data at other levels.

For more documentation, visit
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
for more information.
At this website, you can generate your own API calls using the web interface (click the Download data tab).
At the bottom of the page, click Show API request.
You can copy and paste this into the CDSAPI.py2ju function below.

Some advice so that you can avoid mistakes I have made

- it can take a while for the system to load your data request, especially if you are requesting a lot of data. You can track your requests at https://cds.climate.copernicus.eu/cdsapp#!/yourrequests (look at queued/in progress). Be patient
- Make sure that you set the format to netcdf, otherwise you will get grib data which is a pain to deal with
- Don't modify your raw data, because re-downloading it is annoying (although the second time will be faster -- they temporarily cache things on the servers). Instead, keep your raw data pristine and create a new file with your modifications. I suggest putting these modifications (eg, hourly to daily; spatial averaging; removing anomalies; combining multiple files; etc) in a separate script, perhaps called `process_data.jl`
- If you want to use Python for downloading and/or processing your data, you may. https://docs.xarray.dev/en/stable/ is a great and stable tool with excellent documentation. However, I will not troubleshoot your Python code, so you're on your own.
- Use `abspath` to make sure that your files are saved in the right place. You can also use `joinpath` to make sure that your file paths are correct on different operating systems.
- Make sure you understand this template code! Use AI tools and ask questions as needed.
=#
using CDSAPI
using NCDatasets
using StatsBase: shuffle

# find the "root" directory of your project
HOMEDIR = abspath(dirname(@__FILE__))

"""
    download_single_level_data(year, filename, variable)

Download ERA5 single-level reanalysis data for a specified variable and year.

## Arguments:
- `year`: An integer representing the year for which data is to be downloaded.
- `filename`: A string indicating the name of the file where the data will be saved.
- `variable`: A string representing the variable of interest (e.g., "geopotential").
- `hours` (optional, default `0:23`): A range or array specifying the hours for which data is to be retrieved.
- `resolution` (optional, default `1.0`): A float indicating the spatial resolution of the data in degrees.
- `bbox` (optional, default `[50, -130, 24, -65]`): A vector specifying the bounding box for the spatial domain in the format [North, West, South, East].

## Output:
- If the specified file already exists, a message is printed and `nothing` is returned.
- Otherwise, the function returns the result of the CDSAPI retrieval, which typically saves the data to the specified file.

## Logic Overview:
1. Check if the desired file already exists. If it does, skip the download.
2. Use the CDSAPI to retrieve the specified variable's data for the given year.
3. The data is retrieved for every month, day, and hour within the specified year.
4. The spatial domain is set to cover an area roughly corresponding to the United States with a 1-degree grid resolution.


## Example:
```julia
download_single_level_data(2020, "temperature_2020.nc", "2m_temperature")
```
"""
function download_single_level_data(
    year::Int,
    filename::AbstractString,
    variable::AbstractString;
    hours=0:23,
    resolution=1.0,
    bbox=[50, -130, 24, -65],
)
    if isfile(filename)
        println("File $filename already exists. Skipping download.")
        return nothing
    end

    return CDSAPI.retrieve(
        "reanalysis-era5-single-levels",
        CDSAPI.py2ju("""{
                     "product_type": "reanalysis",
                     "format": "netcdf",
                     "variable": "$variable",
                     "year": "$year",
                     "month": $(["$(lpad(i, 2, '0'))" for i in 1:12]),
                     "day": $(["$(lpad(i, 2, '0'))" for i in 1:31]),
                     "time": $(["$(lpad(hour, 2, '0')):00" for hour in hours]),
                     "area": $bbox,
                     "grid": ["$resolution", "$resolution"],
                     }"""),
        filename,
    )
end

"""
    download_pressure_level_data(year, filename, variable, level)

Download ERA5 pressure-level reanalysis data for a specified variable, year, and pressure level.

## Arguments:
- `year`: An integer representing the year for which data is to be downloaded.
- `filename`: A string indicating the name of the file where the data will be saved.
- `variable`: A string representing the variable of interest (e.g., "geopotential").
- `level`: An integer representing the pressure level (in hPa) of interest (e.g., 500 for 500 hPa).
- `hours` (optional, default `0:23`): A range or array specifying the hours for which data is to be retrieved.
- `resolution` (optional, default `1.0`): A float indicating the spatial resolution of the data in degrees.
- `bbox` (optional, default `[50, -130, 24, -65]`): A vector specifying the bounding box for the spatial domain in the format [North, West, South, East].

## Output:
- If the specified file already exists, a message is printed and `nothing` is returned.
- Otherwise, the function returns the result of the CDSAPI retrieval, which typically saves the data to the specified file.

## Logic Overview:
1. Check if the desired file already exists. If it does, skip the download.
2. Use the CDSAPI to retrieve the specified variable's data for the given year and pressure level.
3. The data is retrieved for every month, day, and hour within the specified year.
4. The spatial domain is set to cover an area roughly corresponding to the United States with a 1-degree grid resolution.

## Example:
```julia
download_pressure_level_data(2020, "geopotential_500hPa_2020.nc", "geopotential", 500)
```
"""
function download_pressure_level_data(
    year::Int,
    filename::AbstractString,
    variable::AbstractString,
    level::Int;
    hours=0:23,
    resolution=1.0,
    bbox=[50, -130, 24, -65],
)
    if isfile(filename)
        println("File $filename already exists. Skipping download.")
        return nothing
    end

    return CDSAPI.retrieve(
        "reanalysis-era5-pressure-levels",
        CDSAPI.py2ju("""{
                     "product_type": "reanalysis",
                     "format": "netcdf",
                     "variable": "$variable",
                     "pressure_level": "$level",
                     "year": "$year",
                     "month": $(["$(lpad(i, 2, '0'))" for i in 1:12]),
                     "day": $(["$(lpad(i, 2, '0'))" for i in 1:31]),
                     "time": $(["$(lpad(hour, 2, '0')):00" for hour in hours]),
                     "area": $bbox,
                     "grid": ["$resolution", "$resolution"],
                     }"""),
        filename,
    )
end

"""
    open_mfdataset(files, variable_name)

Open and concatenate multiple NetCDF files along the time dimension for a specified variable.

## Arguments:
- `files`: A vector of strings, where each string is the path to a NetCDF file.
- `variable_name`: A string representing the name of the variable of interest (e.g., "temperature").

## Output:
- A dictionary containing:
  - The concatenated data for the specified variable.
  - The concatenated time data.
  - Concatenated data for any other 1D coordinates present in the files (excluding the main variable and time).

## Logic Overview:
1. Identify the coordinate names from the first file (excluding the main variable and time).
2. For each file:
   - Extract the data for the main variable, time, and other coordinates.
   - Store this data in separate lists.
3. Pair the main variable data with its corresponding time data.
4. Sort these pairs based on the time data.
5. Concatenate the sorted main variable data and time data.
6. Concatenate the data for the other coordinates.
7. Return a dictionary with the concatenated data, using the actual variable name and coordinate names as keys.

## Example:
```julia
data_dict = open_mfdataset(["file1.nc", "file2.nc", "file3.nc"], "temperature")
```
"""
function open_mfdataset(files::Vector{String}, variable_name::AbstractString)
    # Lists to store variable data, time data, and other coordinate data
    var_data_list = []
    time_data_list = []
    coords_data_dict = Dict()

    # Open the first file to get the coordinate names (excluding time and the main variable)
    ds = Dataset(files[1])
    dimnames = keys(ds.dim)
    coord_names = setdiff(collect(dimnames), [variable_name, "time"])
    close(ds)

    # Initialize lists for each coordinate in coords_data_dict
    for coord in coord_names
        coords_data_dict[coord] = []
    end

    # Open each file, extract data, and store in lists
    for file in files
        ds = Dataset(file)

        # Store variable and time data
        push!(var_data_list, ds[variable_name][:])
        push!(time_data_list, ds["time"][:])

        # Store other coordinate data
        for coord in coord_names
            push!(coords_data_dict[coord], ds[coord][:])
        end

        close(ds)
    end

    # Pair variable data with time data and sort by time
    sorted_pairs = sort(collect(zip(time_data_list, var_data_list)); by=x -> x[1])
    sorted_time_data = [pair[1] for pair in sorted_pairs]
    sorted_var_data = [pair[2] for pair in sorted_pairs]

    # Concatenate sorted data
    concatenated_data_dict = Dict(
        variable_name => vcat(sorted_var_data...), "time" => vcat(sorted_time_data...)
    )

    # Concatenate coordinate data and add to the dictionary
    for coord in coord_names
        concatenated_data_dict[coord] = vcat(coords_data_dict[coord]...)
    end

    return concatenated_data_dict
end

function glob(pattern::AbstractString, dir::AbstractString=".")
    # Convert the glob-like pattern to a regex pattern
    regex_pattern = replace(pattern, "*" => ".*")
    regex_pattern = Regex(regex_pattern)

    # List all files in the directory and filter by the regex pattern
    matching_files = filter(
        filename -> occursin(regex_pattern, filename), readdir(dir; join=true)
    )

    return matching_files
end

function run_demo()

    # the path to the raw data folder
    data_dir = joinpath(HOMEDIR, "data", "raw")

    years = 2019:2020 # example time range
    for year in years

        # Download 2m air temperature for the year 2020
        download_single_level_data.(
            year, joinpath(data_dir, "2m_temperature_$year.nc"), "2m_temperature"
        )

        # Download 500 hPa geopotential for the year 2020
        level = 500
        download_pressure_level_data.(
            year,
            joinpath(data_dir, "$(level)hPa_geopotential_$year.nc"),
            "geopotential",
            level,
        )
    end

    # read in all the 2m temperature data
    fnames = shuffle(glob("2m_temperature", data_dir)) # shuffle -- should work even if out of order
    t2m = open_mfdataset(fnames, "t2m") # we sort based on time, so we don't need to sort here

    # read in all the 500 hPa geopotential data
    fnames = shuffle(glob("500hPa_geopotential", data_dir))
    z500 = open_mfdataset(fnames, "z")

    display(t2m)
    display(z500)

    return nothing
end

run_demo()
