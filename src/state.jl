abstract type State end
abstract type ExplicitState <: State end
abstract type ImplicitState <: State end

function synchronize!(State::S <: ExplicitState, parameters) end

function synchronize!(state::S <: ImplicitState, parameters) end
function jacobian!(solver, state::S <: ImplicitState, parameters, Δt) end
function residual!(state::S <: ImplicitState, Δt) end
function copy_state!(state::S <: ImplicitState) end
function rewind!(state::S <: ImplicitState) end
