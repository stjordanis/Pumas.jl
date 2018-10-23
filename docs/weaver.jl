using Weave: weave
foreach(weave, joinpath("docs", file) for file in readdir("docs") if endswith(file, ".jmd"))
