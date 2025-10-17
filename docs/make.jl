push!(LOAD_PATH, "../src/")

using Documenter, FiniteDiffWENO5

makedocs(
    sitename = "FiniteDiffWENO5",
    remotes = nothing,
    authors = "Hugo Dominguez",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "GettingStarted.md",
        "Background" => "background.md",
        "API" => "API.md",
    ],
)
