
using Gridap
using PkgBenchmark
using DrWatson

target = "raviart_thomas"

results = judge(
  Gridap, 
  BenchmarkConfig(juliacmd = `julia -O3`, id = target),
  BenchmarkConfig(juliacmd = `julia -O3`, id = "master")
)

outfile = projectdir("benchmark/results_$(target).json")
export_markdown(outfile,results)
