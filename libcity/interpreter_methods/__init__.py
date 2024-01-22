def interpreter(model, method='gradient'):
    if method == 'gradient':
        from libcity.interpreter_methods.InterpreterBase import Interpreter
        return Interpreter(model)
    else:
        raise NotImplementedError(f'No such a interpret method of {method}')
