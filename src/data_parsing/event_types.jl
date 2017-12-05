struct Event{T,T2,T3} # Split parameters for dual numbers
  amt::T
  time::T2
  evid::Int8
  cmt::Int
  rate::T
  duration::T
  ss::Int8
  ii::T3
  base_time::T3 # So that this is kept after modifications to duration and rate
  rate_dir::Int8
end

Base.isless(a::Event,b::Event) = isless(a.time,b.time)
Base.isless(a::Event,b::Number) = isless(a.time,b)
Base.isless(a::Number,b::Event) = isless(a,b.time)
