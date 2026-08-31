# (c) 2026 
# Code associated with the paper
# Structural and dynamical properties of hydrous magmatic liquids at high pressure: insights from Raman spectroscopy observations during diamond anvil cell experiments.
# Lauréna Mourand, Clément Ferraina, Salomé Pannefieu, Cyril Aubaud, Charles Le Losq
# Copyrights (c) 2026 Lauréna Mourand, Clément Ferraina, Salomé Pannefieu, Cyril Aubaud, Charles Le Losq
#

## Loading libraries
using Spectra, Statistics, HDF5, Measurements, MultivariateStats
using DataFrames, CSV, LaTeXStrings, RecipesBase, ColorSchemes
import LsqFit
using Plots; gr()
import CairoMakie  # imported (not `using`) to avoid conflicts with Plots.jl
using Typst_jll

# these are relative to 1 CSS px
inch = 96
pt = 4/3
cm = inch / 2.54

col_width = 3.25inch

# Create the folder for saving figures if it doesn't exist
if !isdir("figures")
    mkdir("figures")
end

## Loading the data
data = h5open(joinpath(@__DIR__, "dataset_spectra_T_P.h5"), "r") do f
    Dict(key => read(f[key]) for key in keys(f))
end

x = data["x_axis"]
spectra = data["spectra"]
P = data["P"]
T = data["T"]
exp_nb = data["exp_nb"]
Ruby_P = data["Ruby_P_median"]
Ruby_P_std = data["Ruby_P_std"]
Borate_P = data["Borate_P_median"]
Borate_P_std = data["Borate_P_std"]
Etat = data["Etat"]

# Tg of the starting glass
print("The Deubener calculation of the Tg of the starting glass")
Tg_star = measurement(0.71, 0.1*0.71)
T_GN = measurement(774., 13.)
Tg_G = Tg_star*T_GN - 273
print("\nI have a Tg_star of $(Tg_star), and a $(T_GN), which gives a Tg of $(Tg_G) K")

## Make a first figure summarizing the runs

cmap = ColorSchemes.Spectral
colors = [get(cmap, i / (7 - 1)) for i in 0:6]

fig_pt = CairoMakie.Figure(size=(col_width, col_width), fontsize=12)
ax_pt  = CairoMakie.Axis(fig_pt[1, 1],
    xlabel="Pressure, GPa",
    ylabel="Temperature, °C",
    limits=((2, 15), nothing))

# Position of the annotations
x_ann = [5.5, 3.5, 4.7, 5.6, 10.4, 7.5]
y_ann = [50., 370., 360., 350., 415., 375.]
rotation_ = [0., -55., -55., -55., 0., 0.]


for (count, i) in enumerate([1, 0, 2, 3, 4, 5])
    mask = exp_nb .== i
    x_i  = Float64.(Borate_P[mask])
    y_i  = Float64.(T[mask])
    xe_i = Float64.(Borate_P_std[mask])

    CairoMakie.errorbars!(ax_pt, x_i, y_i, fill(10.0, length(x_i));
        whiskerwidth=4, color=colors[count], linewidth=1)
    CairoMakie.errorbars!(ax_pt, x_i, y_i, xe_i;
        direction=:x, whiskerwidth=4, color=colors[count], linewidth=1)
    CairoMakie.scatter!(ax_pt, x_i, y_i;
        color=colors[count], marker=:circle,
        strokecolor=:black, strokewidth=1)
    CairoMakie.text!(ax_pt, x_ann[count], y_ann[count]; text="Run $(count)", 
                     align=(:center, :center), color=:black, rotation=rotation_[count])
    # CairoMakie.lines!(ax_pt, x_i, y_i;
    #     color=colors[count], linestyle=:dot)
end

# saving...
CairoMakie.save(joinpath(@__DIR__, "figures", "P_T_path.png"), fig_pt, px_per_unit=300/inch)
CairoMakie.save(joinpath(@__DIR__, "figures", "P_T_path.pdf"), fig_pt)
display(fig_pt)

## FITTING PROCEDURE
# We use Spectra.jl to fit the peaks
# First we define the possible peaks
# (peak_type, m_prior, sigma_m_prior, lower_bounds, upper_bounds)
peaks_info = [
    (
        :gaussian, # peak_type
        [0.05, 900.0, 34.0], # m_prior (intensity, position, hwhm)
        [0.1, 10.0, 2.0], # sigma_m_prior
        [0.0, 0.0, 0.0], # lower_bounds
        [Inf, Inf, 70.0], # upper_bounds
    ),
    (
        :gaussian, # peak_type
        [0.25, 950.0, 37.0], # m_prior (intensity, position, hwhm)
        [0.1, 5.0, 2.0], # sigma_m_prior
        [0.0, 0.0, 0.0], # lower_bounds
        [Inf, Inf, 70.0], # upper_bounds
    ),
    (
        :gaussian, # peak_type
        [0.30, 1020.0, 35.0], # m_prior (intensity, position, hwhm)
        [0.2, 5.0, 2.0], # sigma_m_prior
        [0.0, 0.0, 0.0], # lower_bounds
        [Inf, Inf, 70.0], # upper_bounds
    ),
    (
        :pseudovoigt, # peak_type
        [0.70, 1085.0, 27.1, 0.25], # m_prior (intensity, position, hwhm, eta)
        [0.02, 3.0, 2.0, 0.02], # sigma_m_prior
        [0.00, 0.0, 0.0, 0.0], # lower_bounds
        [Inf, Inf, Inf, 0.4], # upper_bounds
    ),
    (
        :gaussian, # peak_type
        [0.30, 1135.0, 35.0], # m_prior (intensity, position, hwhm)
        [0.2, 2.0, 1.5], # sigma_m_prior
        [0.0, 0.0, 0.0], # lower_bounds
        [Inf, Inf, 50.0], # upper_bounds
    ),
]

## Fitting loop

# Get the X values
x_fit = x[x .>820]
y_fit_ = []

# vector to store the results
results = []
ctxs = []

for i = 1:size(spectra, 2)
    # get the portion to fit
    y_fit = spectra[x .>820, i] 

    # area normalize
    #y_fit = (y_fit .- minimum(y_fit)) ./ (maximum(y_fit) - minimum(y_fit))
    y_fit = y_fit ./ maximum(y_fit)

    # linear baseline correction
    roi = [[820.0 830.0]; [1220.0 1300.0]]
    y_fit, y_bas = baseline(x_fit, y_fit, roi=roi, 
    method="polynomial", 
    polynomial_order=1)
    
    global y_fit_ = push!(y_fit)

    # We smooth the spectrum and get an estimate of noise level
    y_smo = smooth(x_fit, y_fit, method="gcvspline");
    estimated_mean_error = sqrt(mean((y_fit .- y_smo).^2))
    estimated_error = estimated_mean_error * ones(size(x_fit));

    ## We can now pass the data and associated errors to prepare_context: 
    ctx = prepare_context(x_fit, y_fit, peaks_info, estimated_error)

    # and we make a first plot
    p = plot_fit(ctx, title="Prior model")

    ## FIT
    result = fit_peaks(ctx, backend=:qGN)

    ## PLOT RESULTS
    p2 = plot_fit(ctx, result.peak_results, title="Fitted model")

    p3 = plot(p, p2, layout=(1, 2))
    
    # save p3
    savefig(p3, joinpath(@__DIR__, "figures", "fit_spectrum_$(i).png"))

    # saving the results
    push!(results, result)
    push!(ctxs, ctx)
end

## Analyse the results
struct PeakData{T}
    position::Vector{T}
    area::Vector{T}
end

peak_names = (:TOH, :Q2, :T2S, :Q3, :Q4) 
peaks = NamedTuple{peak_names}(Tuple(PeakData([], []) for _ in eachindex(peak_names)))

for i = 1:size(spectra, 2)
    println("Spectrum $(i): P=$(P[i]) GPa, T=$(T[i]) K")
    for (j, name) in enumerate(peak_names)
        push!(peaks[name].position, results[i].peak_results[j].params[2])
        push!(peaks[name].area,     results[i].peak_results[j].area)
    end
end

#
# Add dictionaries for symbols and colors for each peak that we will reuse in the plots
# Access examples:
#   peaks.Q2.position  => vector of Q2 positions for all (P,T)
#   peaks.Q2.area      => vector of Q2 areas for all (P,T)
#
cmap_peaks = ColorSchemes.berlin
peak_symbols = Dict(:TOH => :hexagon, :Q2 => :rect, :T2S => :diamond, :Q3 => :utriangle, :Q4 => :pentagon)
peak_colors_scale = [get(cmap_peaks, i / (5 - 1)) for i in 0:5]
peak_colors = Dict(:TOH => peak_colors_scale[1], :Q2 => peak_colors_scale[2], :T2S => peak_colors_scale[3], :Q3 => peak_colors_scale[4], :Q4 => peak_colors_scale[5])

## We remove a two-part linear baseline and area normalize the spectra for PCA analysis of spectra.

sp_ = copy(spectra)
sp_LF = copy(spectra[x .< 830, :])
sp_HF = copy(spectra[x .> 830, :])
Q2_a_ = []
Q3_a_ = []
Q4_a_ = []
Centroid_LF = []
Centroid_HF = []
for i = 1:size(spectra,2)
    
    roi_start = [399 403.]
    roi_stop = [1220. 1225]
    
    # finding the minimum near 800 cm-1
    ymin, imin = findmin(spectra[800 .< x .< 950,i])
    x_min = x[800 .< x .< 900][imin]
 
    # Baseline for the LF part
    Y_bas_, bas_ = baseline(x, spectra[:,i], 
    method="poly", polynomial_order=1, roi=[[400 410.]; [x_min-5 x_min+5.]])
    
    # Baseline for the HF part
    Y_bas2_, bas2_ = baseline(x, spectra[:,i], 
    method="poly", polynomial_order=1, roi=[[x_min-5 x_min+5.]; [1220. 1300]])
    
    # The full baseline is a combination of the two parts
    full_bas_ = [bas_[x .<x_min]; bas2_[x.>=x_min]]
    
    # if you want to plot, activate the following lines
    # p = plot(x, spectra[:,i])
    # plot!(x, full_bas_)
    # display(p)
    # bas_ = rubberband_baseline(x, spectra[:,i])
    
    sp_[:,i] = sp_[:,i] - full_bas_
    
    sp_[:,i] = sp_[:,i]./ sum(sp_[:,i])
    sp_LF[:,i] = sp_[x .< 830, i]/sum(sp_[x .< 830, i])
    sp_HF[:,i] = sp_[x .> 830, i]/sum(sp_[x .> 830, i])
    
    push!(Q2_a_, sum(sp_[ 920 .< x .< 970, i]))
    push!(Q3_a_, sum(sp_[ 1050 .< x .< 1100, i]))
    push!(Q4_a_, sum(sp_[ x .> 1120, i]))
    push!(Centroid_LF, sum(x[x .< 830] .* sp_LF[:,i]))
    push!(Centroid_HF, sum(x[x .> 830] .* sp_HF[:,i]))
    
end


#
# Make the PCA on all spectra
#
y_for_PCA = copy(sp_LF)
nb_cp = 3

M = MultivariateStats.fit(PCA, y_for_PCA; maxoutdim=nb_cp)
y_PCA_for_plot = predict(M, y_for_PCA)

#
# Make a Multidimensional Scaling (MDS) analysis on the spectra
#
M2 = MultivariateStats.fit(MDS, y_for_PCA; maxoutdim=nb_cp, distances=false)
y_MDS_for_plot = predict(M2)

## Tg calculation
# those are the points right before and right after the Tg
Tg_low  = [262., 270., 256., 332., 306.]
Tg_high = [286., 277., 262., 347., 311.]
# Tg_ is placed at the midpoint of the bracket [Tg_low, Tg_high].
# The uncertainty is the half-bracket width (dominant term) combined in quadrature
# with the measurement uncertainty on each bounding temperature (±5°C).
Tg_mid = (Tg_low .+ Tg_high) ./ 2.0
Tg_err = sqrt.( ((Tg_high .- Tg_low) ./ 2.0).^2 .+ 10.0^2 )
Tg_ = measurement.(Tg_mid, Tg_err)
# Crystalization temperature Tc is taken as the temperature of the first point where the sample starts crystallizing.
Tc_ = measurement.([317., 307., 285., 365., 342.], 10)

## First, figure showing Tg vs Tc — Supplementary Figure of the paper
# Tg_ and Tc_ are Measurement types → extract values and uncertainties first
#
# Proper labels for each Cp component in the PCA/MDS plots
labels = ["Cp #1" "Cp #2" "Cp #3" "Cp #4" "Cp #5"]
# associated markers
markers = [:circle :square :diamond :star :hexagon]
# get values for the plot
tg_v = Measurements.value.(Tg_)
tg_e = Measurements.uncertainty.(Tg_)
tc_v = Measurements.value.(Tc_)
tc_e = Measurements.uncertainty.(Tc_)
Fig_TgTc = CairoMakie.Figure(size=(col_width, col_width), fontsize=12)
TgTc_Ax = CairoMakie.Axis(Fig_TgTc[1, 1],
    xlabel=CairoMakie.rich("T", CairoMakie.subscript("g"), ", °C"),
    ylabel=CairoMakie.rich("T", CairoMakie.subscript("ctx"), ", °C"))
CairoMakie.errorbars!(TgTc_Ax, tg_v, tc_v, tg_e; direction=:x, whiskerwidth=4, color=:black, linewidth=1)
CairoMakie.errorbars!(TgTc_Ax, tg_v, tc_v, tc_e; whiskerwidth=4, color=:black, linewidth=1)
CairoMakie.scatter!(TgTc_Ax, tg_v, tc_v; color=:black)
CairoMakie.resize_to_layout!(Fig_TgTc)
CairoMakie.save(joinpath(@__DIR__, "figures", "Tg_Tc.pdf"), Fig_TgTc)
CairoMakie.save(joinpath(@__DIR__, "figures", "Tg_Tc.png"), Fig_TgTc, px_per_unit=300/inch)
display(Fig_TgTc)

##  Figures showing P_borate, P_ruby and MDS components vs T for each experiment
#
# Sample n_colors evenly spaced colors from the colormap
cmap = ColorSchemes.batlow  # or any other colormap you prefer
colors = [get(cmap, i / (4 - 1)) for i in 0:3]

# ── CairoMakie loop ────────────────────────────────────────────────────────
for (count, i) in enumerate([0, 2, 3, 4, 5])

    end_idx = (i == 2) ? 1 : 0

    Pressure_min = minimum(Float64.(P[exp_nb .== i]))
    Pressure_max = maximum(Float64.(P[exp_nb .== i]))
    Tmax         = maximum(Float64.(T[exp_nb .== i]))
    X_temp       = Float64.(T[exp_nb .== i][1:end-end_idx])

    x_lims = (minimum(X_temp) - 5.0, maximum(X_temp) + 5.0)
    p_lims = (Pressure_min - 1.0, Pressure_max + 1.0)

    ruby_p   = Float64.(Ruby_P[exp_nb .== i][1:end-end_idx])
    ruby_std = Float64.(Ruby_P_std[exp_nb .== i][1:end-end_idx])
    bor_p    = Float64.(Borate_P[exp_nb .== i][1:end-end_idx])
    bor_std  = Float64.(Borate_P_std[exp_nb .== i][1:end-end_idx])

    y_PCA_ = y_PCA_for_plot[:, exp_nb .== i][1:nb_cp, 1:end-end_idx]'
    pca_ylims = (minimum(y_PCA_) - 0.003, maximum(y_PCA_) + 0.003)

    Tg_val = Measurements.value(Tg_[count])
    Tc_val = Measurements.value(Tc_[count])

    fig_exp = CairoMakie.Figure(size=(2*col_width, 3.0inch), fontsize=12)

    # a) Ruby pressure vs T
    ax_ruby = CairoMakie.Axis(fig_exp[1, 1];
        xlabel="Temperature, °C", ylabel="Pressure from ruby, GPa",
        limits=(x_lims, p_lims), title="a)")
    CairoMakie.vspan!(ax_ruby, Tc_val, Tmax + 5; color=(:red,    0.3))
    CairoMakie.vspan!(ax_ruby, Tg_val, Tc_val;   color=(:orange, 0.2))
    CairoMakie.errorbars!(ax_ruby, X_temp, ruby_p, ruby_std; whiskerwidth=4, linewidth=1, color=:black)
    CairoMakie.scatter!(ax_ruby, X_temp, ruby_p; color=:black, markersize=10)

    # b) Borate pressure vs T
    ax_bor = CairoMakie.Axis(fig_exp[1, 2];
        xlabel="Temperature, °C", ylabel="Pressure from borate, GPa",
        limits=(x_lims, p_lims), title="b)")
    CairoMakie.vspan!(ax_bor, Tc_val, Tmax + 5; color=(:red,    0.3))
    CairoMakie.vspan!(ax_bor, Tg_val, Tc_val;   color=(:orange, 0.2))
    CairoMakie.errorbars!(ax_bor, X_temp, bor_p, bor_std; whiskerwidth=4, linewidth=1, color=:black)
    CairoMakie.scatter!(ax_bor, X_temp, bor_p; color=:black, markersize=10)

    # c) MDS components vs T
    makie_markers = [:circle, :rect, :diamond]
    ax_pca = CairoMakie.Axis(fig_exp[1, 3];
        xlabel="Temperature, °C", ylabel="Component value, a.u.",
        limits=(x_lims, pca_ylims), title="c)")
    CairoMakie.vspan!(ax_pca, Tc_val, Tmax + 5; color=(:red,    0.3))
    CairoMakie.vspan!(ax_pca, Tg_val, Tc_val;   color=(:orange, 0.2))
    for j in 1:nb_cp
        y_lims_ = (minimum(y_PCA_[:, j]) - 0.003, maximum(y_PCA_[:, j]) + 0.003)
        CairoMakie.scatter!(ax_pca, X_temp, y_PCA_[:, j];
            marker=makie_markers[j], label="Cp #$j", markersize=10, strokecolor=:black, strokewidth=1)
    end

    # Annotations (top-left of each panel)
    # for (k, ax, ytop) in zip(1:3,
    #                           (ax_ruby, ax_bor, ax_pca),
    #                           (p_lims[2], p_lims[2], pca_ylims[2]))
    #     CairoMakie.text!(ax, x_lims[1] + 5, ytop;
    #         text="$(Char(96+k)))", align=(:left, :top))
    # end

    CairoMakie.save(joinpath(@__DIR__, "figures", "P_T_Cp_exp_$(i).pdf"), fig_exp)
    CairoMakie.save(joinpath(@__DIR__, "figures", "P_T_Cp_exp_$(i).png"), fig_exp, px_per_unit=300/inch)
    display(fig_exp)
end

## Plot Qn areas

sum_ = peaks.Q2.area+peaks.Q3.area+peaks.Q4.area+peaks.TOH.area+peaks.T2S.area
TOH_ = peaks.TOH.area./sum_
Q2_ = peaks.Q2.area./sum_
Q3_ = peaks.Q3.area./sum_
Q4_ = peaks.Q4.area./sum_
T2S_ = peaks.T2S.area./sum_

# List of experiments to use
exp_nb_list = [0, 2, 3, 4, 5]

# ── CairoMakie version ──────────────────────────────────────────────────────
function plot_Qn_makie!(ax, T_exp, Tg_val, Tc_val, Q2, T2S, Q3, Q4, TOH)
    Tmax = maximum(T_exp)
    # Shaded regions (drawn first, behind data)
    CairoMakie.vspan!(ax, Tc_val, Tmax + 5; color=(:red,    0.3))
    CairoMakie.vspan!(ax, Tg_val, Tc_val;   color=(:orange, 0.2))
    # Scatter + errorbars for each Qn species
    for (vals, sym, col) in [
        (Q2,  peak_symbols[:Q2],  peak_colors[:Q2]),
        (T2S, peak_symbols[:T2S], peak_colors[:T2S]),
        (Q3,  peak_symbols[:Q3],  peak_colors[:Q3]),
        (Q4,  peak_symbols[:Q4],  peak_colors[:Q4]),
        (TOH, peak_symbols[:TOH], peak_colors[:TOH]),
    ]
        v = Measurements.value.(vals)
        e = Measurements.uncertainty.(vals)
        CairoMakie.errorbars!(ax, T_exp, v, e; whiskerwidth=4, color=col, linewidth=1)
        CairoMakie.scatter!(ax, T_exp, v; marker=sym, color=col, markersize=10, strokecolor=:black, strokewidth=1)
    end
end

fig_Qn    = CairoMakie.Figure(size=(2*col_width, 3*col_width), fontsize=12)
positions = [(1,1),(1,2),(2,1),(2,2),(3,1)]

for (count, i) in enumerate([0, 2, 3, 4, 5])
    mask   = exp_nb .== i
    T_exp  = Float64.(T[mask])
    Tg_val = Measurements.value(Tg_[count])
    Tc_val = Measurements.value(Tc_[count])
    P_med  = round(median(Float64.(Borate_P[mask])); digits=1)
    pos    = positions[count]

    ax = CairoMakie.Axis(fig_Qn[pos[1], pos[2]];
        xlabel = (count in [4, 5]) ? "Temperature, °C" : "",
        ylabel = (count in [1, 3, 5]) ? "Relative area" : "",
        limits = ((minimum(T_exp)-10, maximum(T_exp)+10), (0.0, 0.4)),
        title="$(Char(96+count))) Run $(count+1), P ≈ $P_med GPa",
        width = 2inch, height = 2inch)

    plot_Qn_makie!(ax, T_exp, Tg_val, Tc_val,
        Q2_[mask], T2S_[mask], Q3_[mask], Q4_[mask], TOH_[mask])

end

# Legend in bottom-right cell (markers + colored region patches)
leg_entries = [
    [CairoMakie.MarkerElement(marker=peak_symbols[:TOH], color=peak_colors[:TOH], markersize=10, strokecolor=:black, strokewidth=1)],
    [CairoMakie.MarkerElement(marker=peak_symbols[:Q2],  color=peak_colors[:Q2],  markersize=10, strokecolor=:black, strokewidth=1)],
    [CairoMakie.MarkerElement(marker=peak_symbols[:T2S], color=peak_colors[:T2S], markersize=10, strokecolor=:black, strokewidth=1)],
    [CairoMakie.MarkerElement(marker=peak_symbols[:Q3],  color=peak_colors[:Q3],  markersize=10, strokecolor=:black, strokewidth=1)],
    [CairoMakie.MarkerElement(marker=peak_symbols[:Q4],  color=peak_colors[:Q4],  markersize=10, strokecolor=:black, strokewidth=1)],
    [CairoMakie.PolyElement(color=(:orange, 0.2), strokecolor=:transparent)],
    [CairoMakie.PolyElement(color=(:red,    0.3), strokecolor=:transparent)],
]
leg_labels = ["TOH", 
CairoMakie.rich("Q", CairoMakie.superscript("2")), 
CairoMakie.rich("T", CairoMakie.subscript("2S")), 
CairoMakie.rich("Q", CairoMakie.superscript("3")), 
CairoMakie.rich("Q", CairoMakie.superscript("4")), 
"Melt", 
"Melt + crystals"]
CairoMakie.Legend(fig_Qn[3, 2], leg_entries, leg_labels, framevisible=false)
CairoMakie.resize_to_layout!(fig_Qn)
CairoMakie.save(joinpath(@__DIR__, "figures", "Qn_areas.pdf"), fig_Qn)
CairoMakie.save(joinpath(@__DIR__, "figures", "Qn_areas.png"), fig_Qn, px_per_unit=300/inch)
CairoMakie.display(fig_Qn)

## Now we do a figure comparing start glass, melt and quench.
# Spectre 10 => 25 °C, 5 GPa, Start Glass
# Spectre 21 => 197 °C, 4.8 GPa, Quench
# Spectre 19 => 302 °C, 4.5 GPa, Melt

# Get the 25 °C / 1 atm spectrum and treat it
df = DataFrame
sp_25C_1atm = CSV.read(joinpath(@__DIR__, "ANN_5prctH2O_LF.asc"), df, header=["x", "y"], types=[Float64, Float64])

# Convert nm to cm
sp_25C_1atm[:,1] = nm_to_invcm(sp_25C_1atm[:,1], laser_nm=532.0)

# despiking
sp_25C_1atm[:,2] = despiking(sp_25C_1atm[:,1], sp_25C_1atm[:,2])

# Baseline removal
sp_25C_1atm[:,2], bas_ = baseline(sp_25C_1atm[:,1], sp_25C_1atm[:,2], roi=[1220 1300.], method="poly", polynomial_order=0)

# Long correction
~, sp_25C_1atm[:,2], ~ = tlcorrection(Matrix(sp_25C_1atm), 25.0, 532.0)

# calculate centroid at LF
centroid_glass_25C = sum(sp_25C_1atm[sp_25C_1atm[:,1] .< 830,1] .* sp_25C_1atm[sp_25C_1atm[:,1] .< 830,2]) / sum(sp_25C_1atm[sp_25C_1atm[:,1] .< 830,2])
print("Centroid of 25 °C / 1 atm spectrum at LF: $(centroid_glass_25C) cm⁻¹\n")
#
# Figure with all interesting spectra
#

fig = CairoMakie.Figure(size=(col_width, col_width), fontsize=12)  #
ax = CairoMakie.Axis(fig[1, 1],
    xlabel=CairoMakie.rich("Raman shift, cm",CairoMakie.superscript("-1")),
    ylabel="Intensity, area normalised",
    limits=(400, 1300, nothing, nothing)
)

CairoMakie.lines!(ax, sp_25C_1atm[:,1], sp_25C_1atm[:,2], color=:black, linewidth=1.0)
CairoMakie.lines!(ax, x, spectra[:,10] .+ 0.0015, color=:purple, linewidth=1.0)
CairoMakie.lines!(ax, x, spectra[:,21] .+ 0.003, color=:red, linewidth=1.0)
CairoMakie.lines!(ax, x, spectra[:,19] .+ 0.0045, color=:orange, linewidth=1.0)

CairoMakie.text!(ax, 800, 0.0012, text="1 atm\n25 °C", align=(:center, :top), fontsize=8)
CairoMakie.text!(ax, 820, 0.0029, text="≈5 GPa\n25 °C", align=(:center, :top), fontsize=8)
CairoMakie.text!(ax, 840, 0.0044, text="≈5 GPa\n302 °C", align=(:center, :top), fontsize=8)
CairoMakie.text!(ax, 830, 0.0060, text="≈5 GPa\n197 °C", align=(:center, :top), fontsize=8)

CairoMakie.text!(ax, 1240, 0.0003, text="Glass", align=(:center, :center), fontsize=8)
CairoMakie.text!(ax, 1240, 0.0018, text="Glass", align=(:center, :center), fontsize=8)
CairoMakie.text!(ax, 1240, 0.0033, text="Melt", align=(:center, :center), fontsize=8)
CairoMakie.text!(ax, 1240, 0.0052, text="HP\nGlass", align=(:center, :center), fontsize=8)

CairoMakie.lines!(ax, [520., 550.], [0.0016, 0.0032], color=:black, linestyle=:dot)
CairoMakie.lines!(ax, [600., 635., 650., 650.], [0.0016, 0.0032, 0.005, 0.0063], color=:black, linestyle=:dot)
CairoMakie.lines!(ax, [950., 950.], [0.0013, 0.0058], color=:black, linestyle=:dot)
CairoMakie.lines!(ax, [1070., 1080, 1085, 1085.], [0.0051, 0.0054, 0.0062, 0.0075], color=:black, linestyle=:dot)

CairoMakie.save(joinpath(@__DIR__, "figures", "Glass_Melt_comparison.pdf"), fig)
CairoMakie.save(joinpath(@__DIR__, "figures", "Glass_Melt_comparison.png"), fig, px_per_unit=300/96)
CairoMakie.display(fig)

## PLOT HP SPECTRA
#
#
# Now we select all spectra that correspond to the "Melt" state and plots them together,
# with a color gradient corresponding to the pressure.
mask_melt = Etat .== "Melt"
P_melt = Float64.(Borate_P[mask_melt])
P_melt_std = Float64.(Borate_P_std[mask_melt])
T_melt = Float64.(T[mask_melt])
exp_nb_melt = exp_nb[mask_melt]

# get the pressure at Tg for each experiment where there is melt
P_melt_Tg = [P_melt[exp_nb_melt .==i][1] for i in [0, 2, 3, 4, 5]]
P_melt_Tg_std = [P_melt_std[exp_nb_melt .==i][1] for i in [0, 2, 3, 4, 5]]

# sorting the pressure and corresponding spectra
sort_idx = sortperm(P_melt)
P_melt_sorted = P_melt[sort_idx]
P_melt_std_sorted = P_melt_std[sort_idx]
T_melt_sorted = T_melt[sort_idx]
spectra_sorted = spectra[:, mask_melt][:, sort_idx]
spectra_LF_sorted = sp_LF[:, mask_melt][:, sort_idx]
spectra_HF_sorted = sp_HF[:, mask_melt][:, sort_idx]
exp_nb_melt_sorted = exp_nb_melt[sort_idx]

# results of peak fitting
ctxs_sorted = ctxs[mask_melt][sort_idx]
results_sorted = results[mask_melt][sort_idx]
TOH_sorted = TOH_[mask_melt][sort_idx]
Q2_sorted = Q2_[mask_melt][sort_idx]
T2S_sorted = T2S_[mask_melt][sort_idx]
Q3_sorted = Q3_[mask_melt][sort_idx]
Q4_sorted = Q4_[mask_melt][sort_idx]
Centroid_LF_sorted = Centroid_LF[mask_melt][sort_idx]
Centroid_HF_sorted = Centroid_HF[mask_melt][sort_idx]

TOH_position_sorted = peaks.TOH.position[mask_melt][sort_idx] 
Q2_position_sorted = peaks.Q2.position[mask_melt][sort_idx]
T2S_position_sorted = peaks.T2S.position[mask_melt][sort_idx] 
Q3_position_sorted = peaks.Q3.position[mask_melt][sort_idx]
Q4_position_sorted = peaks.Q4.position[mask_melt][sort_idx]

# normalising pressure for good colors
P_norm = (P_melt_sorted .- minimum(P_melt_sorted)) ./ (maximum(P_melt_sorted) - minimum(P_melt_sorted))
colors_melt = [get(cmap, p) for p in P_norm]

## MAKE THE P-T phase plot
#
#
# we stored above the pressure of the melt at Tg.
# add the value at 1 bar
P_melt_Tg = push!(P_melt_Tg, 0.0)
P_melt_Tg_std = push!(P_melt_Tg_std, 0.0)

# add Tg at 1 bar in Tg_
Tg_ = push!(Tg_, 277.) # 10% uncertainty

# sort both arrays using increasing P
sorted_indices = sortperm(P_melt_Tg)
P_Tg_sorted = P_melt_Tg[sorted_indices]
P_Tg_sorted_std = P_melt_Tg_std[sorted_indices]
Tg_sorted = Tg_[sorted_indices]

fig = CairoMakie.Figure(size=(5.0inch, 3.0inch), fontsize=12)
ax = CairoMakie.Axis(fig[1, 1], xlabel="Pressure (GPa)", ylabel="Temperature (°C)")

# Glass
mask = Etat .== "Verre"
CairoMakie.errorbars!(ax, Borate_P[mask], T[mask], Borate_P_std[mask]; direction=:x, color=:grey, linewidth=1)
CairoMakie.errorbars!(ax, Borate_P[mask], T[mask], fill(10., count(mask)); direction=:y, color=:grey, linewidth=1)
CairoMakie.scatter!(ax, Borate_P[mask], T[mask], color=:grey, marker=:rect, label="Glass", strokecolor=:black, strokewidth=1)

# Melt
mask = Etat .== "Melt"
CairoMakie.errorbars!(ax, Borate_P[mask], T[mask], Borate_P_std[mask]; direction=:x, color=:orange, linewidth=1)
CairoMakie.errorbars!(ax, Borate_P[mask], T[mask], fill(10., count(mask)); direction=:y, color=:orange, linewidth=1)
CairoMakie.scatter!(ax, Borate_P[mask], T[mask], color=:orange, marker=:circle, label="Melt", strokecolor=:black, strokewidth=1)

# Melt+Ctx
mask = Etat .== "Melt+Ctx"
CairoMakie.errorbars!(ax, Borate_P[mask], T[mask], Borate_P_std[mask]; direction=:x, color=:red, linewidth=1)
CairoMakie.errorbars!(ax, Borate_P[mask], T[mask], fill(10., count(mask)); direction=:y, color=:red, linewidth=1)
CairoMakie.scatter!(ax, Borate_P[mask], T[mask], color=:red, marker=:diamond, label="Melt+Ctx", strokecolor=:black, strokewidth=1)

# add Tg at 1 atm, estimated around 277 °C
CairoMakie.errorbars!(ax, [0.0], [277.], [56.]; color=:red, linewidth=1, direction=:y) 
CairoMakie.scatter!(ax, [0.0], [277.]; color=:red, marker=:star5, markersize=25, 
                    label=CairoMakie.rich("T", CairoMakie.subscript("g"), " at 1 atm"))

# add Tg line
CairoMakie.lines!(ax, P_Tg_sorted, Tg_sorted; color=:red, linestyle=:dash, 
                    label=CairoMakie.rich("T", CairoMakie.subscript("g")))

CairoMakie.Legend(fig[1,2], ax, framevisible=false)

CairoMakie.save(joinpath(@__DIR__, "figures", "P_T_diagram.pdf"), fig)
CairoMakie.save(joinpath(@__DIR__, "figures", "P_T_diagram.png"), fig, px_per_unit=300/96)

CairoMakie.display(fig)

## Figure showing the evolution of the LF silicate signal centroid with pressure
#
#
# First we get glass data (25 °C)
Centroid_glass = Centroid_LF[T .== 25.0]
P_glass = Borate_P[T .== 25.0]
P_glass_std = Borate_P_std[T .== 25.0]

# append the value at 1 atm
Centroid_glass = push!(Centroid_glass, centroid_glass_25C)
P_glass = push!(P_glass, 0.0)
P_glass_std = push!(P_glass_std, 0.0)


fig = CairoMakie.Figure(size=(col_width, col_width), fontsize=12)
ax = CairoMakie.Axis(fig[1, 1],
    xlabel="Pressure, GPa",
    ylabel=CairoMakie.rich("LF silicate signal centroid, cm", CairoMakie.superscript("-1")),
    limits=(0, 14, 560, 670)
)

# Ligne verticale vers 6.5 GPa
#CairoMakie.vlines!(ax, [6.5], color=:black, linestyle=:dot, linewidth=10, alpha=0.5)

# Zones colorées avant/après 6.5 GPa
CairoMakie.vspan!(ax, [0.0], [6.0], color=(:blue, 0.1))
CairoMakie.vspan!(ax, [6.0], [7.0], color=(:black, 0.2))
CairoMakie.vspan!(ax, [7.0], [14.0], color=(:green, 0.1))

# plot the melt data points on top
CairoMakie.errorbars!(ax, P_melt_sorted, Centroid_LF_sorted, P_melt_std_sorted; direction=:x, color=:black, linewidth=1)
CairoMakie.scatter!(ax, P_melt_sorted, Centroid_LF_sorted, marker=:circle, color=:red, markersize=10,
                    strokewidth=1, strokecolor=:black)

# add the glass data points on top
CairoMakie.errorbars!(ax, P_glass, Centroid_glass, P_glass_std; direction=:x, color=:black, linewidth=1)
CairoMakie.scatter!(ax, P_glass, Centroid_glass, marker=:utriangle, color=:purple, markersize=10, strokewidth=1, strokecolor=:black)

## add fitted trend lines for melt and glass

# fit a poly(2) and plot it for glass
fn(x, p) = @. p[1] + p[2] * x + p[3] * x.^2
prob = LsqFit.curve_fit(fn, P_glass, Centroid_glass, [1.0, 1.0, 1.0])
x_fit = range(0, stop=6.5, length=100)
CairoMakie.lines!(ax, x_fit, fn(x_fit, prob.param); 
                  color=:purple, linestyle=:dot, linewidth=5, alpha=0.5, label="Glass fit")

# add a sigmoid+ linear trend for the melt
fn2(x, p) = @. p[1] + p[2] / (1 + exp(-p[3] * (x - p[4]))) + p[5] * x
prob2 = LsqFit.curve_fit(fn2, P_melt_sorted, Centroid_LF_sorted, [610, 30., 1., 6.5, 1.0])
x_fit2 = range(0, stop=14, length=100)
CairoMakie.lines!(ax, x_fit2, fn2(x_fit2, prob2.param); 
                  color=:red, linestyle=:dash, linewidth=5, alpha=0.5, label="Melt fit")

# Annotations texte (désactivées, comme dans l'original)
CairoMakie.text!(ax, 3.0, 659, text="Network\ncompaction", align=(:center, :center), color=:black)
CairoMakie.text!(ax, 7.0+3.5, 659, text="Coordination\nincrease", align=(:center, :center), color=:black)
CairoMakie.text!(ax, 5.0, 635, text="Melt", align=(:center, :center), color=:red)
CairoMakie.text!(ax, 4.1, 595, text="Glass", align=(:center, :center), color=:purple)
#text!(ax, 10.25, 655, text="High pressure region", fontsize=8, align=(:center, :center), color=:black)
#scatter!(ax, [0.0], [561.0], marker=:hexagon, color=:black, markersize=6)

CairoMakie.save(joinpath(@__DIR__, "figures", "LF_Centroid.pdf"), fig)
CairoMakie.save(joinpath(@__DIR__, "figures", "LF_Centroid.png"), fig, px_per_unit=300/inch)

display(fig)

## Figure showing the evolution of the Qn peak positions with pressure
#
#

fig = CairoMakie.Figure(size=(col_width, col_width), fontsize=12)
ax = CairoMakie.Axis(fig[1, 1], xlabel="Pressure, GPa", ylabel=CairoMakie.rich("Frequency, cm", CairoMakie.superscript("-1")))

CairoMakie.errorbars!(ax, P_melt_sorted, Measurements.value.(TOH_position_sorted), Measurements.uncertainty.(TOH_position_sorted); color=peak_colors[:TOH])
CairoMakie.scatter!(ax, P_melt_sorted, Measurements.value.(TOH_position_sorted), marker=peak_symbols[:TOH], color=peak_colors[:TOH], label=L"TOH", strokecolor=:black, strokewidth=1)

CairoMakie.errorbars!(ax, P_melt_sorted, Measurements.value.(Q2_position_sorted), Measurements.uncertainty.(Q2_position_sorted); color=peak_colors[:Q2])
CairoMakie.scatter!(ax, P_melt_sorted, Measurements.value.(Q2_position_sorted), marker=peak_symbols[:Q2], color=peak_colors[:Q2], label=L"$Q^{2}$", strokecolor=:black, strokewidth=1)

CairoMakie.errorbars!(ax, P_melt_sorted, Measurements.value.(T2S_position_sorted), Measurements.uncertainty.(T2S_position_sorted); color=peak_colors[:T2S])
CairoMakie.scatter!(ax, P_melt_sorted, Measurements.value.(T2S_position_sorted), marker=peak_symbols[:T2S], color=peak_colors[:T2S], label=L"$T_{2S}$", strokecolor=:black, strokewidth=1)

CairoMakie.errorbars!(ax, P_melt_sorted, Measurements.value.(Q3_position_sorted), Measurements.uncertainty.(Q3_position_sorted); color=peak_colors[:Q3])
CairoMakie.scatter!(ax, P_melt_sorted, Measurements.value.(Q3_position_sorted), marker=peak_symbols[:Q3], color=peak_colors[:Q3], label=L"$Q^{3}$", strokecolor=:black, strokewidth=1)

CairoMakie.errorbars!(ax, P_melt_sorted, Measurements.value.(Q4_position_sorted), Measurements.uncertainty.(Q4_position_sorted); color=peak_colors[:Q4])
CairoMakie.scatter!(ax, P_melt_sorted, Measurements.value.(Q4_position_sorted), marker=peak_symbols[:Q4], color=peak_colors[:Q4], label=L"$Q^{4}$", strokecolor=:black, strokewidth=1)

CairoMakie.text!(ax, 9, 900, text="TOH")
CairoMakie.text!(ax, 9, 960, text=CairoMakie.rich("Q", CairoMakie.superscript("2")))
CairoMakie.text!(ax, 9, 1030, text=CairoMakie.rich("T", CairoMakie.subscript("2S")))
CairoMakie.text!(ax, 9, 1090, text=CairoMakie.rich("Q", CairoMakie.superscript("3")))
CairoMakie.text!(ax, 9, 1130, text=CairoMakie.rich("Q", CairoMakie.superscript("4")))

CairoMakie.save(joinpath(@__DIR__, "figures", "Peak_position_vs_P.pdf"), fig)
CairoMakie.save(joinpath(@__DIR__, "figures", "Peak_position_vs_P.png"), fig, px_per_unit=300/96)

display(fig)

## QN UNIT CALCULATIONS AND PLOT
#
# CALCULATION OF QN UNIT QUANTITIES
#
Qn_sorted_sum = Q2_sorted + Q3_sorted + Q4_sorted #+ T2S_sorted
Q2_ult = Q2_sorted./Qn_sorted_sum .* 0.95
Q3_ult = (Q3_sorted./Qn_sorted_sum ).* 0.95
Q4_ult = 1.0 .- Q2_ult .- Q3_ult
NBOT = 2 .* Q2_ult + Q3_ult

#
# Figure showing the evolution of the Qn unit quantities with pressure
#

#
# PANNEAU 1 : QUANTITÉS D'UNITÉS QN
#
fig = CairoMakie.Figure(size=(2*col_width, col_width), fontsize=12)  # taille approx pour combiner deux panneaux, ajuste si besoin

ax1 = CairoMakie.Axis(fig[1, 1],
    xlabel="Pressure, GPa", ylabel="Peak area (melt only)",
    title="a)",
    limits=(nothing, nothing, 0.0, 0.45)
)

# plot error bars on y
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(TOH_sorted), Measurements.uncertainty.(TOH_sorted),color=peak_colors[:TOH], linewidth=1)
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(Q2_sorted), Measurements.uncertainty.(Q2_sorted),color=peak_colors[:Q2],linewidth=1)
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(T2S_sorted), Measurements.uncertainty.(T2S_sorted),color=peak_colors[:T2S],linewidth=1)
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(Q3_sorted), Measurements.uncertainty.(Q3_sorted),color=peak_colors[:Q3],linewidth=1)
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(Q4_sorted), Measurements.uncertainty.(Q4_sorted),color=peak_colors[:Q4],linewidth=1)

# Plot error bars on x
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(TOH_sorted), P_melt_std_sorted,color=peak_colors[:TOH], linewidth=1, direction=:x)
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(Q2_sorted), P_melt_std_sorted,color=peak_colors[:Q2],linewidth=1, direction=:x)
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(T2S_sorted), P_melt_std_sorted,color=peak_colors[:T2S],linewidth=1, direction=:x)
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(Q3_sorted), P_melt_std_sorted,color=peak_colors[:Q3],linewidth=1, direction=:x)
CairoMakie.errorbars!(ax1, P_melt_sorted, Measurements.value.(Q4_sorted), P_melt_std_sorted,color=peak_colors[:Q4],linewidth=1, direction=:x)

# plot markers for values
CairoMakie.scatter!(ax1, P_melt_sorted, TOH_sorted, marker=peak_symbols[:TOH], color=peak_colors[:TOH], label="TOH peak", strokecolor=:black, strokewidth=1)
CairoMakie.scatter!(ax1, P_melt_sorted, Q2_sorted, marker=peak_symbols[:Q2], color=peak_colors[:Q2], label=L"$Q^2$", strokecolor=:black, strokewidth=1)
CairoMakie.scatter!(ax1, P_melt_sorted, T2S_sorted, marker=peak_symbols[:T2S], color=peak_colors[:T2S], label=L"$T_{2S}$", strokecolor=:black, strokewidth=1)
CairoMakie.scatter!(ax1, P_melt_sorted, Q3_sorted, marker=peak_symbols[:Q3], color=peak_colors[:Q3], label=L"$Q^3$", strokecolor=:black, strokewidth=1)
CairoMakie.scatter!(ax1, P_melt_sorted, Q4_sorted, marker=peak_symbols[:Q4], color=peak_colors[:Q4], label=L"$Q^4$", strokecolor=:black, strokewidth=1)

# annotate

CairoMakie.text!(ax1, 8, 0.04, text="TOH", color=peak_colors[:TOH])
CairoMakie.text!(ax1, 8, 0.14, text=CairoMakie.rich("Q", CairoMakie.superscript("2")), color=peak_colors[:Q2])
CairoMakie.text!(ax1, 8, 0.26, text=CairoMakie.rich("T", CairoMakie.subscript("2S")), color=peak_colors[:T2S])
CairoMakie.text!(ax1, 8, 0.33, text=CairoMakie.rich("Q", CairoMakie.superscript("3")), color=peak_colors[:Q3])
CairoMakie.text!(ax1, 8, 0.20, text=CairoMakie.rich("Q", CairoMakie.superscript("4")), color=peak_colors[:Q4])

#
# PANNEAU 2 : Q2/Q3 vs pression, coloré par température
#
ax3 = CairoMakie.Axis(fig[1, 2],
    xlabel="Pressure, GPa", ylabel=CairoMakie.rich("Q", CairoMakie.superscript("2"), "/Q", CairoMakie.superscript("3"), " area ratio (melt only)"),
    title="b)",
    limits=(2, 13, 0.2, 0.8)
)

CairoMakie.errorbars!(ax3, P_melt_sorted, Measurements.value.(Q2_sorted ./ Q3_sorted), Measurements.uncertainty.(Q2_sorted ./ Q3_sorted); direction=:y, color=:grey)
CairoMakie.errorbars!(ax3, P_melt_sorted, Measurements.value.(Q2_sorted ./ Q3_sorted), P_melt_std_sorted; direction=:x, color=:grey)

sc = CairoMakie.scatter!(ax3, P_melt_sorted, Measurements.value.(Q2_sorted ./ Q3_sorted),
    color=T_melt_sorted, colormap=:magma, marker=:rect)

CairoMakie.Colorbar(fig[1, 3], sc, label="Temperature, °C")

CairoMakie.lines!(ax3, [2., 6.5], [0.57, 0.57], color=:red, linestyle=:dash, linewidth=2.0)
CairoMakie.lines!(ax3, [6.5, 12.0], [0.48, 0.48], color=:red, linestyle=:dash, linewidth=2.0)

CairoMakie.save(joinpath(@__DIR__, "figures", "Qn_melt_area.pdf"), fig)
CairoMakie.save(joinpath(@__DIR__, "figures", "Qn_melt_area.png"), fig, px_per_unit=300/inch)

display(fig)

## LOW FREQUENCY COMPARISON
#
#
x_LF = x[x .< 830]
p = plot([NaN],[NaN],
        xlabel=L"Raman shift, cm$^{-1}$",
        ylabel="Normalised intensity, a.u.",
        label=false,
        figsize=(3.22*200,3.22*200),
        )
for (count,i) in enumerate([1, 10, 15, 20])
    label_ = "P = $(round(P_melt_sorted[i]; digits=2)) / T = $(round(T_melt_sorted[i]; digits=2))"
    plot!(x_LF, spectra_LF_sorted[:,i]./sum(spectra_LF_sorted[:,i]), 
    color = colors_melt[i],
    label=label_,legend=:topleft)
end
savefig(joinpath(@__DIR__, "figures", "LF_melt_spectra.pdf"))
display(p)

## HIGH FREQUENCY COMPARISON
#
#
x_HF = x[x .> 830]
p = plot([NaN],[NaN],
        xlabel=L"Raman shift, cm$^{-1}$",
        ylabel="Normalised intensity, a.u.",
        label=false,
        figsize=(3.22*200,3.22*200),
        )
for (count,i) in enumerate([1, 10, 15, 20])
    label_ = "P = $(round(P_melt_sorted[i]; digits=2)) / T = $(round(T_melt_sorted[i]; digits=2))"
    plot!(x_HF, spectra_HF_sorted[:,i]./sum(spectra_HF_sorted[:,i]), 
    color = colors_melt[i],
    label=label_,legend=:topleft)
end
savefig(joinpath(@__DIR__, "figures", "HF_melt_spectra.pdf"))
display(p)

## PLOT THE FITS AT LOW AND HIGH PRESSURE, SIMILAR T.
#
#
p_ = []
for i in [1, 20]
    p = plot_fit(results_sorted[i].context, results_sorted[i].peak_results, 
         title="P ≈ $(round(P_melt_sorted[i]; digits=1)) GPa / T ≈ $(round(T_melt_sorted[i])) °C")
    push!(p_, p)
end
PP_ = plot(p_[1], p_[2])
savefig(joinpath(@__DIR__, "figures", "HF_melt_fit_comparison.pdf"))
savefig(joinpath(@__DIR__, "figures", "HF_melt_fit_comparison.png"))
display(PP_)

## Make a figure showing the evolution of Tg for those three compositions
#
#
# Data from Bagdassarov 2004
Albite_Tg = Dict("P_GPa" => [0.1, 0.5, 1.0, 3.0, 6.0],
                 "Tg" => Measurement.([
                    683.0 ± 4.0,
                    690.0 ± 3.0,
                    676.0 ± 5.0,
                    663.0 ± 5.0,
                    630.0 ± 15.0
                 ]))
Albite_Tg["dTg"] = Albite_Tg["Tg"][:] .- Albite_Tg["Tg"][1]
Anorthite_Tg = Dict("P_GPa" => [0.1, 0.3, 0.6, 3.0, 4.5, 6.0],
                 "Tg" => Measurement.([
                    844.0 ± 5.0,
                    847.0 ± 6.0,
                    863.0 ± 6.0,
                    859.0 ± 6.0,
                    873.0 ± 6.0,
                    881.0 ± 15
                 ]))
Anorthite_Tg["dTg"] = Anorthite_Tg["Tg"][:] .- Anorthite_Tg["Tg"][1]
ANN_Tg = Dict("P_GPa" => P_Tg_sorted,
              "Tg" => Tg_sorted,
              "dTg" => Tg_sorted[:] .- Tg_sorted[1])

# equation from Russell 2024 for peridotite
A_R2024 = -5.36
Bo_R2024 = 5408.5
B1_R2024 = -2553.0
B2_R2024 = 84.21
Co_R2024 = 432.08
C1_R2024 = 2.69
C2_R2024 = -525.4 
P_peridotite= LinRange(0.0001, 15.0, 30)
B = Bo_R2024 .+ B1_R2024 .* 0.029 .+ B2_R2024 .* (P_peridotite .- 0.0001)
C = Co_R2024 + 86.7*C1_R2024 .+ C2_R2024*0.029.^0.5 
Tg_peridotite = B/(12.0-A_R2024) .+ C 

fig = CairoMakie.Figure(size=(5.0inch, 3.0inch), fontsize=12)
ax = CairoMakie.Axis(fig[1, 1],
    xlabel="Pressure, GPa", ylabel=CairoMakie.rich("ΔT", CairoMakie.subscript("g"), ", K"),
    limits=(0.0, 15, nothing, nothing)
)

# ANN, this study

CairoMakie.errorbars!(ax, ANN_Tg["P_GPa"], Measurements.value.(ANN_Tg["dTg"]), Measurements.uncertainty.(ANN_Tg["dTg"]); 
                    direction=:y, color=:red, whiskerwidth = 4, linewidth=1)
CairoMakie.errorbars!(ax, ANN_Tg["P_GPa"], Measurements.value.(ANN_Tg["dTg"]), P_Tg_sorted_std, 
                    direction = :x, color=:red, whiskerwidth = 4, linewidth=1)
CairoMakie.scatter!(ax, ANN_Tg["P_GPa"], Measurements.value.(ANN_Tg["dTg"]), 
                    marker=:rect, markersize=12, color=:red, label="ANN, this study", strokecolor=:black, strokewidth=1)

# Anorthite, B2004
CairoMakie.errorbars!(ax, Anorthite_Tg["P_GPa"], Measurements.value.(Anorthite_Tg["dTg"]), Measurements.uncertainty.(Anorthite_Tg["dTg"]); 
                    direction=:y, color=:black, linewidth=1)
CairoMakie.scatter!(ax, Anorthite_Tg["P_GPa"], Measurements.value.(Anorthite_Tg["dTg"]), 
                    marker=:circle, markersize=12, color=:black, label="Anorthite, B2004", strokecolor=:black, strokewidth=1)

# Albite, B2004
CairoMakie.errorbars!(ax, Albite_Tg["P_GPa"], Measurements.value.(Albite_Tg["dTg"]), Measurements.uncertainty.(Albite_Tg["dTg"]); 
                    direction=:y, color=:blue, linewidth=1)
CairoMakie.scatter!(ax, Albite_Tg["P_GPa"], Measurements.value.(Albite_Tg["dTg"]), 
                    marker=:diamond, markersize=12, color=:blue, label="Albite, B2004", strokecolor=:black, strokewidth=1)

# Peridotite, R2024 (pas de barres d'erreur dans l'original)
#lines!(ax, P_peridotite, Tg_peridotite .- Tg_peridotite[1], color=:orange, linestyle=:dash, label="Peridotite, R2024")


CairoMakie.lines!(ax, P_peridotite, Tg_peridotite .- Tg_peridotite[1], color=:orange, linestyle=:dash, label="Peridotite, R2024")

CairoMakie.Legend(fig[1,2], ax, framevisible=false)

CairoMakie.save(joinpath(@__DIR__, "figures", "Tg_evolution.pdf"), fig)
CairoMakie.save(joinpath(@__DIR__, "figures", "Tg_evolution.png"), fig, px_per_unit=300/inch)

fig