## Special overloads to avoid ambiguity errors
eval(generate_code_2_to_1(:Base, ^, Integer))
eval(generate_code_2_to_1(:Base, ^, Rational))
eval(generate_code_2_to_1(:Base, ^, Irrational{:ℯ}))
eval(generate_code_2_to_1(:Base, isless, AbstractFloat))
