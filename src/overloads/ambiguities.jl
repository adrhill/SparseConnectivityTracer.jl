## Special overloads to avoid ambiguity errors
eval(generate_code_2_to_1_typed(:Base, ^, Integer))
eval(generate_code_2_to_1_typed(:Base, ^, Rational))
eval(generate_code_2_to_1_typed(:Base, ^, Irrational{:ℯ}))
eval(generate_code_2_to_1_typed(:Base, isless, AbstractFloat))
