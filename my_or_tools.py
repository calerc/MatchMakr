
'''
    OrSolver
    a CpSolver that returns status without require access to private
    attribute __solution
'''
class ORSolver(cp_model.CpSolver):

    def __init__(self):
        super(ORSolver, self).__init__()
        self.status = None
        
    def SolveWithSolutionCallback(self, model, callback):
        """Solves a problem and pass each solution found to the callback."""
        self.__solution = (
            pywrapsat.SatHelper.SolveWithParametersAndSolutionCallback(
                model.ModelProto(), self.parameters, callback))
        
        status = self.__solution.status
        status = self.StatusName(status)
        self.status = status
        
        return status
    
    def return_status(self):
        return self.status
    
    def Value(self, expression):
        return EvaluateLinearExpression(expression, self.__solution)
    

'''
    VarArrayAndObjectiveSolutionPrinter
    callback printer object for ortools solver
'''
class VarArrayAndObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):

    ''' Print intermediate solutions. '''

    def __init__(self, match_maker):

        cp_model.CpSolverSolutionCallback.__init__(self)

        self.match_maker = match_maker
        self.variables = match_maker.interview

        self.CHECK_MATCHES = self.match_maker.CHECK_MATCHES
        self.CHECK_FREQUENCY = self.match_maker.CHECK_FREQUENCY

        if self.CHECK_MATCHES:
            self.last_stud_percent = np.zeros(
                (1, self.match_maker.NUM_PREFERENCES_2_CHECK))
            self.last_faculty_percent = np.zeros(
                (1, self.match_maker.NUM_PREFERENCES_2_CHECK))

        self.__solution_count = 0

    def on_solution_callback(self):

        # Get sizes
        num_faculty = self.match_maker.num_faculty
        num_students = self.match_maker.num_students
        num_interviews = self.match_maker.NUM_INTERVIEWS

        # Print objective value
        text = 'Solution %i' % self.__solution_count
        try:
            self.match_maker.print(text)
        except Exception as e:
            print('Error!')
            print(e)
        text = '  objective value = %i' % self.ObjectiveValue()
        try:
            self.match_maker.print(text)
        except Exception as e:
            print('Error!')
            print(e)

        # Determine what matches were made
        if self.CHECK_MATCHES and (
                self.__solution_count %
                self.CHECK_FREQUENCY == 0):

            values = np.empty((num_students, num_faculty, num_interviews))
            for p in range(num_faculty):
                for s in range(num_students):
                    for i in range(num_interviews):
                        the_key = (p, s, i)
                        the_variable = self.variables[the_key]
                        values[s, p, i] = self.Value(the_variable)

            values = np.asarray(values)

            # Determine number of matches made for preferences
            matches = np.sum(values, axis=2)
            self.match_maker.check_preferences(matches)

        self.__solution_count += 1

    def solution_count(self):
        return self.__solution_count