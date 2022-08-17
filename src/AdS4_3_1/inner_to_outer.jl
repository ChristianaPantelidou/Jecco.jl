
@inline S_inner_to_outer(S_in, u, xi) =
    1/u + xi + u*u*S_in

@inline F_inner_to_outer(F_in, u) = u * F_in

@inline A_inner_to_outer(A_in, u, xi) =
    1/(u*u) + 2*xi/u + xi*xi + u * A_in



@inline S_u_inner_to_outer(S_u_in, S_in, u, xi) =
    -1/(u*u)+ 2 *u * S_in + u*u * S_u_in

@inline F_u_inner_to_outer(F_u_in, F_in, u) = F_in + u* F_u_in

@inline A_u_inner_to_outer(A_u_in, A_in, u, xi) =
    -2/(u*u*u) - 2*xi/(u*u) + A_in + u* A_u_in



@inline Sd_inner_to_outer(Sd_in, u, xi) =
    1/(2*u*u) + xi/u + xi*xi/2  + u* Sd_in

@inline Bd_inner_to_outer(Bd_in, u) = u*u*Bd_in


