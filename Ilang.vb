
<Serializable>
Public Class iLangModel
    Private LanguageModel As NgramLanguageModel
    Private Attend As FeedForwardNetwork
    Private csize As Integer

    Private iVocabulary As Vocabulary
    Private ReadOnly Property EncodingVocabulary As Vocabulary
        Get
            Return iVocabulary
        End Get
    End Property

    ''' <summary>
    ''' Create New Model
    ''' </summary>
    Public Sub New()
        csize = 1
        iVocabulary = New Vocabulary
        LanguageModel = New NgramLanguageModel(2)
        Attend = New FeedForwardNetwork(csize, 8, 1)

    End Sub
    ''' <summary>
    ''' Can be set with a known vocabulary
    ''' </summary>
    ''' <param name="model"></param>
    Public Sub New(model As iLangModel)
        Me.iVocabulary = model.EncodingVocabulary
    End Sub
    ''' <summary>
    ''' This input is encoded as a single value, 
    ''' So Char by Char , Word by Word , 
    ''' Sent by Sent is decided outside the object
    ''' </summary>
    ''' <param name="uInputWord"></param>
    ''' <returns></returns>
    Public Function EncodeInput(ByRef uInputWord As String) As Integer


        If EncodingVocabulary.CheckExists(uInputWord) = False Then
            LanguageModel.AddDocument(uInputWord)
            iVocabulary.ADD_NEW(uInputWord, LanguageModel.LookupNgram(uInputWord))
            Return iVocabulary.LOOKUP(uInputWord)
        Else
            Return iVocabulary.LOOKUP(uInputWord)
        End If

    End Function
    ''' <summary>
    ''' look up the value of the token provided 
    ''' </summary>
    ''' <param name="Query"></param>
    ''' <returns></returns>
    Public Function DecodeInput(ByRef Query As Integer) As String
        Return iVocabulary.LOOKUP(Query)
    End Function
    Public Function forward(inputSequence As List(Of List(Of Double))) As List(Of List(Of Double))
        'Here we want to see what the output is without positional encoding
        Return ApplyFeedForwardNN(ApplyMuliHeadedAttention(inputSequence, 3, inputSequence.Count))
    End Function
    Public Sub Train(inputs As List(Of List(Of Double)), targets As List(Of List(Of Double)), epochs As Integer, learningRate As Double)
        csize = inputs.ElementAt(0).Count
        Attend.Train(ApplyMuliHeadedAttention(inputs, 3, inputs.Count), targets, epochs, learningRate)
    End Sub
    Public Function ApplyMuliHeadedAttention(inputSequence As List(Of List(Of Double)), numHeads As Integer, headSize As Integer, Optional Masked As Boolean = False) As List(Of List(Of Double))
        Dim Attend As New MultiHeadedAttention(numHeads, headSize)
        Return Attend.Forward(inputSequence, Masked)
    End Function
    Public Function PredictNext_LangModel(ByRef Userinput As String) As String
        'Add Dynamic to corpus
        Dim words = Split(Userinput, " ").ToList
        For Each word In words
            EncodeInput(word)
        Next
        'Load Question into vocabulary(as Question)
        EncodeInput(Userinput)
        'Return Prediction Word Or Sentence?
        Return LanguageModel.PredictNextWord(Userinput)
    End Function
    Public Function PredictNext_Transformer(ByRef Userinput As String) As String
        EncodeInput(Userinput)
        Dim words = Split(Userinput, " ").ToList
        'Load Positional Encodings
        Dim Encoder As New PositionalEncoding(8, 8, iVocabulary.VocabList)
        Dim InputSequence = Encoder.Encode(words)
        'Transform
        Dim Ouput = ApplyFeedForwardNN(ApplyMuliHeadedAttention(InputSequence, 3, InputSequence.Count))
        'decode Positions
        Dim decoder As New PositionalDecoder(8, 8, iVocabulary.VocabList)
        'Build Decoded Output 
        Dim str As String = ""
        For Each item In decoder.Decode(Ouput)
            str &= item & " "
        Next
        Return str

    End Function
    Public Function ApplyFeedForwardNN(inputSequence As List(Of List(Of Double))) As List(Of List(Of Double))

        csize = inputSequence.ElementAt(0).Count

        Return Attend.Forward(inputSequence)
    End Function
    Public Shared Function FlattenList(lst As List(Of List(Of Double))) As List(Of Integer)
        Dim iFlat As New List(Of Integer)
        For Each i In lst
            For Each item In i
                iFlat.Add(item)
            Next
        Next
        Return iFlat
    End Function

End Class
<Serializable>
Public Class PositionalEncoding
    Private ReadOnly encodingMatrix As List(Of List(Of Double))
    Private Vocabulary As New List(Of String)
    Public Sub New(maxLength As Integer, embeddingSize As Integer, ByRef vocab As List(Of String))
        encodingMatrix = New List(Of List(Of Double))
        Vocabulary = vocab
        ' Create the encoding matrix
        For pos As Integer = 0 To maxLength - 1
            Dim encodingRow As List(Of Double) = New List(Of Double)()

            For i As Integer = 0 To embeddingSize - 1
                Dim angle As Double = pos / Math.Pow(10000, (2 * i) / embeddingSize)
                encodingRow.Add(Math.Sin(angle))
                encodingRow.Add(Math.Cos(angle))
            Next

            encodingMatrix.Add(encodingRow)
        Next
    End Sub

    Public Function Encode(inputTokens As List(Of String)) As List(Of List(Of Double))
        Dim encodedInputs As List(Of List(Of Double)) = New List(Of List(Of Double))()

        For pos As Integer = 0 To inputTokens.Count - 1
            Dim token As String = inputTokens(pos)
            Dim tokenEncoding As List(Of Double) = New List(Of Double)()

            ' Retrieve the positional encoding for the token
            tokenEncoding = encodingMatrix(pos)

            encodedInputs.Add(tokenEncoding)
        Next

        Return encodedInputs
    End Function

    Public Function iEncode(inputTokens As List(Of String)) As List(Of List(Of Double))
        Dim encodedInputs As List(Of List(Of Double)) = New List(Of List(Of Double))()

        For Each token As String In inputTokens
            Dim tokenEncoding As List(Of Double) = New List(Of Double)()

            ' Find the index of the token in the vocabulary
            ' For simplicity, let's assume a fixed vocabulary
            Dim tokenIndex As Integer = GetTokenIndex(token)

            ' Retrieve the positional encoding for the token
            If tokenIndex >= 0 Then
                tokenEncoding = encodingMatrix(tokenIndex)
            Else
                ' Handle unknown tokens if necessary
            End If

            encodedInputs.Add(tokenEncoding)
        Next

        Return encodedInputs
    End Function
    Private Function GetTokenIndex(token As String) As Integer
        ' Retrieve the index of the token in the vocabulary
        ' For simplicity, let's assume a fixed vocabulary
        Dim vocabulary As List(Of String) = GetVocabulary()
        Return vocabulary.IndexOf(token)
    End Function

    Private Function GetVocabulary() As List(Of String)
        ' Return the vocabulary list
        ' Modify this function as per your specific vocabulary
        Return Vocabulary
    End Function
End Class
<Serializable>
Public Class MultiHeadedAttention
    Private Shared irand As Random = New Random()
    Private ReadOnly headSize As Integer
    Private ReadOnly numHeads As Integer

    Public Sub New(numHeads As Integer, headSize As Integer)
        Me.numHeads = numHeads
        Me.headSize = headSize
        Randomize()

    End Sub
    Private Shared Function GetRandomWeight() As Double

        Return irand.NextDouble()
    End Function
    Private Shared Function InitializeWeights(rows As Integer, cols As Integer) As List(Of List(Of Double))
        Dim weights As List(Of List(Of Double)) = New List(Of List(Of Double))
        irand.NextDouble()

        For i As Integer = 0 To rows - 1
            Dim rowWeights As List(Of Double) = New List(Of Double)()
            irand.NextDouble()
            For j As Integer = 0 To cols - 1
                rowWeights.Add(GetRandomWeight)
            Next

            weights.Add(rowWeights)
        Next

        Return weights
    End Function

    Private Function iMaskedAttention(query As List(Of List(Of Double)), key As List(Of List(Of Double)), value As List(Of List(Of Double))) As List(Of List(Of Double))
        Dim attendedFeatures As List(Of List(Of Double)) = New List(Of List(Of Double))

        For Each queryVector As List(Of Double) In query
            Dim weightedValues As List(Of Double) = New List(Of Double)

            For Each keyVector As List(Of Double) In key
                Dim attentionValue As Double = 0.0

                For i As Integer = 0 To headSize - 1
                    attentionValue += queryVector(i) * keyVector(i)
                Next

                ' Apply masking by setting attention value to 0 for padding vectors
                If keyVector.All(Function(x) x = 0) Then
                    attentionValue = 0.0
                End If

                weightedValues.Add(attentionValue)
            Next

            attendedFeatures.Add(weightedValues)
        Next

        Return attendedFeatures
    End Function
    Private Function iAttention(query As List(Of List(Of Double)), key As List(Of List(Of Double)), value As List(Of List(Of Double))) As List(Of List(Of Double))
        Dim attendedFeatures As List(Of List(Of Double)) = New List(Of List(Of Double))

        For Each queryVector As List(Of Double) In query
            Dim weightedValues As List(Of Double) = New List(Of Double)

            For Each keyVector As List(Of Double) In key
                Dim attentionValue As Double = 0.0

                For i As Integer = 0 To headSize - 1
                    attentionValue += queryVector(i) * keyVector(i)
                Next

                weightedValues.Add(attentionValue)
            Next

            attendedFeatures.Add(weightedValues)
        Next

        Return attendedFeatures
    End Function
    Public Function LinearTransformation(inputSequence As List(Of List(Of Double))) As List(Of List(Of Double))
        Dim transformedSequence As List(Of List(Of Double)) = New List(Of List(Of Double))
        irand.NextDouble()
        Dim outputWeight As List(Of List(Of Double)) = InitializeWeights(numHeads * headSize, headSize)

        For Each vector As List(Of Double) In inputSequence
            Dim transformedVector As List(Of Double) = New List(Of Double)()

            For j As Integer = 0 To headSize - 1
                Dim transformedValue As Double = 0.0

                For k As Integer = 0 To numHeads - 1
                    transformedValue += vector(j + k * headSize) * outputWeight(j + k * headSize)(j)
                Next

                transformedVector.Add(transformedValue)
            Next

            transformedSequence.Add(transformedVector)
        Next

        Return transformedSequence
    End Function
    Public Function SplitByHead(inputSequence As List(Of List(Of Double)), numHeads As Integer) As List(Of List(Of List(Of Double)))
        Dim splitInput As List(Of List(Of List(Of Double))) = New List(Of List(Of List(Of Double)))(numHeads)

        For i As Integer = 0 To numHeads - 1
            Dim headSequence As List(Of List(Of Double)) = New List(Of List(Of Double))()

            For Each vector As List(Of Double) In inputSequence
                Dim headVector As List(Of Double) = vector.GetRange(i * headSize, headSize)
                headSequence.Add(headVector)
            Next

            splitInput.Add(headSequence)
        Next

        Return splitInput
    End Function
    Public Function ConcatenateHeads(headOutputs As List(Of List(Of List(Of Double)))) As List(Of List(Of Double))
        Dim concatenatedOutput As List(Of List(Of Double)) = New List(Of List(Of Double))()

        For i As Integer = 0 To headOutputs(0).Count - 1
            Dim concatenatedVector As List(Of Double) = New List(Of Double)()

            For Each headOutput As List(Of List(Of Double)) In headOutputs
                concatenatedVector.AddRange(headOutput(i))
            Next

            concatenatedOutput.Add(concatenatedVector)
        Next

        Return concatenatedOutput
    End Function
    Public Function Transform(query As List(Of List(Of Double)), key As List(Of List(Of Double)), value As List(Of List(Of Double)), Optional useMaskedAttention As Boolean = False) As List(Of List(Of Double))
        ' Split the query, key, and value into multiple heads
        Dim splitQuery = SplitByHead(query, numHeads)
        Dim splitKey = SplitByHead(key, numHeads)
        Dim splitValue = SplitByHead(value, numHeads)

        ' Apply attention mechanism for each head
        Dim headOutputs As List(Of List(Of List(Of Double))) = New List(Of List(Of List(Of Double)))(numHeads)
        For i As Integer = 0 To numHeads - 1
            Dim q As List(Of List(Of Double)) = splitQuery(i)
            Dim k As List(Of List(Of Double)) = splitKey(i)
            Dim v As List(Of List(Of Double)) = splitValue(i)

            Dim headOutput As List(Of List(Of Double))
            If useMaskedAttention Then
                headOutput = iMaskedAttention(q, k, v)
            Else
                headOutput = iAttention(q, k, v)
            End If

            headOutputs.Add(headOutput)
        Next

        ' Concatenate the head outputs
        Dim concatenatedOutput As List(Of List(Of Double)) = ConcatenateHeads(headOutputs)

        ' Apply linear transformation
        Dim output As List(Of List(Of Double)) = LinearTransformation(concatenatedOutput)

        Return output
    End Function


    Public Function Attention(inputSequence As List(Of List(Of Double)), inputVector As List(Of Double)) As List(Of Double)
        Dim weightedValues As List(Of Double) = New List(Of Double)()

        For Each sequenceVector As List(Of Double) In inputSequence
            Dim attentionValue As Double = 0.0

            For i As Integer = 0 To headSize - 1
                attentionValue += inputVector(i) * sequenceVector(i)
            Next

            weightedValues.Add(attentionValue)
        Next

        Return weightedValues
    End Function
    Public Function MaskedAttention(inputSequence As List(Of List(Of Double)), inputVector As List(Of Double)) As List(Of Double)
        Dim weightedValues As List(Of Double) = New List(Of Double)

        For Each sequenceVector As List(Of Double) In inputSequence
            Dim attentionValue As Double = 1

            For i As Integer = 0 To headSize - 1
                attentionValue += inputVector(i) * sequenceVector(i)
            Next

            ' Apply masking by setting attention value to 0 for padding vectors
            If sequenceVector.All(Function(x) x = 0) Then
                attentionValue = 0
            End If

            weightedValues.Add(attentionValue)
        Next

        Return weightedValues
    End Function
    Public Function Forward(inputSequence As List(Of List(Of Double)), Optional useMaskedAttention As Boolean = False) As List(Of List(Of Double))
        Dim attendedFeatures As List(Of List(Of Double)) = New List(Of List(Of Double))()

        For Each inputVector As List(Of Double) In inputSequence
            Dim attendedVector As List(Of Double)
            If useMaskedAttention Then
                attendedVector = MaskedAttention(inputSequence, inputVector)
            Else
                attendedVector = Attention(inputSequence, inputVector)
            End If

            attendedFeatures.Add(attendedVector)
        Next

        Return attendedFeatures
    End Function



End Class
<Serializable>
Public Class FeedForwardNetwork
    Public Enum Activation
        ReLU
        Sigmoid
        Tanh
    End Enum
    Private ReadOnly hiddenSize As Integer
    Private ReadOnly hiddenWeights As List(Of List(Of Double))
    Private ReadOnly inputSize As Integer
    Private ReadOnly layerNorm1 As LayerNormalization
    Private ReadOnly layerNorm2 As LayerNormalization
    Private ReadOnly outputSize As Integer
    Private ReadOnly outputWeights As List(Of List(Of Double))

    Private rand As Random = New Random()
    Private outputGradients As List(Of List(Of Double))
    Private hiddenGradients As List(Of List(Of Double))
    Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer)
        Me.inputSize = inputSize
        Me.hiddenSize = hiddenSize
        Me.outputSize = outputSize
        Randomize()
        Me.hiddenWeights = InitializeWeights(inputSize, hiddenSize)
        Me.outputWeights = InitializeWeights(hiddenSize, outputSize)

        ' Initialize layer normalization objects
        Me.layerNorm1 = New LayerNormalization(hiddenSize)
        Me.layerNorm2 = New LayerNormalization(outputSize)

        ' Initialize positional encoding object
        '   Me.positionalEncoder = New PositionalEncoderFF(hiddenSize)
        outputGradients = New List(Of List(Of Double))
        hiddenGradients = New List(Of List(Of Double))
    End Sub


    ''' <summary>
    ''' Trains the feed-forward neural network using gradient descent optimization.
    ''' </summary>
    ''' <param name="inputs">The input training data.</param>
    ''' <param name="targets">The target training data.</param>
    ''' <param name="epochs">The number of training epochs.</param>
    ''' <param name="learningRate">The learning rate for gradient descent.</param>
    Public Sub Train(inputs As List(Of List(Of Double)), targets As List(Of List(Of Double)), epochs As Integer, learningRate As Double)
        For epoch As Integer = 1 To epochs
            Dim lossSum As Double = 0.0

            For i As Integer = 0 To inputs.Count - 1
                Dim inputVector As List(Of Double) = inputs(i)
                Dim targetVector As List(Of Double) = targets(i)

                ' Forward pass to compute the predicted output
                Dim outputVector As List(Of Double) = Forward(inputs)(i)

                ' Compute the loss (e.g., mean squared error)
                Dim loss As Double = ComputeLoss(outputVector, targetVector)
                lossSum += loss

                ' Backpropagation to compute gradients
                Backpropagation(inputVector, outputVector, targetVector)

                ' Update the weights using gradient descent
                UpdateWeights(learningRate)
            Next

            ' Compute the average loss for the epoch
            Dim averageLoss As Double = lossSum / inputs.Count

            ' Print the average loss for monitoring
            Console.WriteLine("Epoch {0}: Average Loss = {1}", epoch, averageLoss)
        Next
    End Sub
    ''' <summary>
    ''' Computes the loss between the predicted output and the target output.
    ''' </summary>
    ''' <param name="outputVector">The predicted output vector.</param>
    ''' <param name="targetVector">The target output vector.</param>
    ''' <returns>The loss value.</returns>
    Private Function ComputeLoss(outputVector As List(Of Double), targetVector As List(Of Double)) As Double
        Dim loss As Double = 0.0
        Dim n As Integer = outputVector.Count

        For i As Integer = 0 To n - 1
            loss += (outputVector(i) - targetVector(i)) ^ 2
        Next

        loss /= n

        Return loss
    End Function

    ''' <summary>
    ''' Performs backpropagation to compute the gradients.
    ''' </summary>
    ''' <param name="inputVector">The input vector.</param>
    ''' <param name="outputVector">The predicted output vector.</param>
    ''' <param name="targetVector">The target output vector.</param>
    Private Sub Backpropagation(inputVector As List(Of Double), outputVector As List(Of Double), targetVector As List(Of Double))
        ' Compute the gradient of the output layer
        outputGradients = New List(Of List(Of Double))()
        Dim outputDelta As List(Of Double) = New List(Of Double)()

        For i As Integer = 0 To outputSize - 1
            Dim derivative As Double = outputVector(i) - targetVector(i)
            outputDelta.Add(derivative)
        Next

        ' Compute the gradient of the hidden layer
        hiddenGradients = New List(Of List(Of Double))()
        Dim hiddenDelta As List(Of Double) = New List(Of Double)()

        For i As Integer = 0 To hiddenSize - 1
            Dim derivative As Double = HiddenActivationDerivative(inputVector, i) * WeightedSum(outputDelta, outputWeights, i)
            hiddenDelta.Add(derivative)
        Next

        outputGradients.Add(outputDelta)
        hiddenGradients.Add(hiddenDelta)
    End Sub
    ''' <summary>
    ''' Computes the weighted sum of the inputs using the specified weights and index.
    ''' </summary>
    ''' <param name="inputs">The input vector.</param>
    ''' <param name="weights">The weight matrix.</param>
    ''' <param name="index">The index of the neuron.</param>
    ''' <returns>The weighted sum.</returns>
    Private Function WeightedSum(inputs As List(Of Double), weights As List(Of List(Of Double)), index As Integer) As Double
        Dim sum As Double = 0.0

        For i As Integer = 0 To inputs.Count - 1
            sum += inputs(i) * weights(i)(index)
        Next

        Return sum
    End Function

    ''' <summary>
    ''' Updates the weights of the neural network using gradient descent.
    ''' </summary>
    ''' <param name="learningRate">The learning rate for gradient descent.</param>
    Private Sub UpdateWeights(learningRate As Double)
        ' Update the weights between the hidden and output layers
        For i As Integer = 0 To hiddenSize - 1
            For j As Integer = 0 To outputSize - 1
                Dim weightChange As Double = -learningRate * outputGradients(0)(j)
                outputWeights(i)(j) += weightChange
            Next
        Next

        ' Update the weights between the input and hidden layers
        For i As Integer = 0 To inputSize - 1
            For j As Integer = 0 To hiddenSize - 1
                Dim weightChange As Double = -learningRate * hiddenGradients(0)(j)
                hiddenWeights(i)(j) += weightChange
            Next
        Next
    End Sub

    ''' <summary>
    ''' Computes the derivative of the activation function used in the hidden layer.
    ''' </summary>
    ''' <param name="inputVector">The input vector.</param>
    ''' <param name="index">The index of the neuron.</param>
    ''' <returns>The derivative value.</returns>
    Private Function HiddenActivationDerivative(inputVector As List(Of Double), index As Integer) As Double
        Dim sum As Double = 0.0

        For i As Integer = 0 To inputSize - 1
            sum += inputVector(i) * hiddenWeights(i)(index)
        Next

        Dim output As Double = HiddenActivation(sum)
        Return output * (1 - output)
    End Function
    ''' <summary>
    ''' Applies the activation function to the hidden layer outputs.
    ''' </summary>
    ''' <param name="input">The input value.</param>
    ''' <returns>The activated value.</returns>
    Private Function HiddenActivation(input As Double) As Double
        ' Use the sigmoid function as the activation function for the hidden layer
        Return 1.0 / (1.0 + Math.Exp(-input))
    End Function
    ''' <summary>
    ''' Applies the activation function to the output layer outputs.
    ''' </summary>
    ''' <param name="input">The input value.</param>
    ''' <returns>The activated value.</returns>
    Private Function OutputActivation(input As Double) As Double
        ' Use the identity function as the activation function for the output layer
        Return input
    End Function



    Public Function Forward(inputs As List(Of List(Of Double))) As List(Of List(Of Double))
        Dim hiddenOutputs As List(Of List(Of Double)) = LinearTransformation(inputs, hiddenWeights, hiddenSize)

        ' Apply layer normalization and residual connections
        Dim norm1Outputs As List(Of List(Of Double)) = layerNorm1.Normalize(hiddenOutputs)
        Dim residual1Outputs As List(Of List(Of Double)) = AddResidualConnections(hiddenOutputs, norm1Outputs)
        're-inject input focusing(update memory)
        residual1Outputs = AddResidualConnections(residual1Outputs, inputs)
        Dim activatedOutputs As List(Of List(Of Double)) = ApplyActivation(residual1Outputs, Activation.ReLU)
        Dim finalOutputs As List(Of List(Of Double)) = LinearTransformation(activatedOutputs, outputWeights, outputSize)
        rand.NextDouble()
        ' Apply layer normalization and residual connections
        Dim norm2Outputs As List(Of List(Of Double)) = layerNorm2.Normalize(finalOutputs)
        Dim residual2Outputs As List(Of List(Of Double)) = AddResidualConnections(finalOutputs, norm2Outputs)
        Return residual2Outputs
    End Function

    Private Function AddResidualConnections(inputs As List(Of List(Of Double)), outputs As List(Of List(Of Double))) As List(Of List(Of Double))
        Dim residualOutputs As List(Of List(Of Double)) = New List(Of List(Of Double))(inputs.Count)

        For i As Integer = 0 To inputs.Count - 1
            Dim residualVector As List(Of Double) = New List(Of Double)(inputs(i).Count)

            For j As Integer = 0 To inputs(i).Count - 1
                residualVector.Add(inputs(i)(j) + outputs(i)(j))
                rand.NextDouble()
            Next

            residualOutputs.Add(residualVector)
        Next

        Return residualOutputs
    End Function

    Private Function ApplyActivation(inputs As List(Of List(Of Double)), activation As Activation) As List(Of List(Of Double))
        Dim activatedOutputs As List(Of List(Of Double)) = New List(Of List(Of Double))(inputs.Count)

        For Each inputVector As List(Of Double) In inputs
            Dim activatedVector As List(Of Double) = New List(Of Double)(inputVector.Count)

            For Each inputVal As Double In inputVector
                Select Case activation
                    Case Activation.ReLU
                        activatedVector.Add(Math.Max(0, inputVal))
                    Case Activation.Sigmoid
                        activatedVector.Add(1 / (1 + Math.Exp(-inputVal)))
                    Case Activation.Tanh
                        activatedVector.Add(Math.Tanh(inputVal))
                End Select
            Next

            activatedOutputs.Add(activatedVector)
        Next

        Return activatedOutputs
    End Function

    Private Function GetRandomWeight() As Double
        Return rand.NextDouble()
    End Function

    Private Function InitializeWeights(inputSize As Integer, outputSize As Integer) As List(Of List(Of Double))
        Dim weights As List(Of List(Of Double)) = New List(Of List(Of Double))(inputSize)
        rand.NextDouble()
        For i As Integer = 0 To inputSize - 1
            Dim weightVector As List(Of Double) = New List(Of Double)(outputSize)
            rand.NextDouble()
            For j As Integer = 0 To outputSize - 1
                weightVector.Add(GetRandomWeight())
            Next

            weights.Add(weightVector)
        Next

        Return weights
    End Function

    Private Function LinearTransformation(inputs As List(Of List(Of Double)), weights As List(Of List(Of Double)), outputSize As Integer) As List(Of List(Of Double))
        Dim output As List(Of List(Of Double)) = New List(Of List(Of Double))(inputs.Count)

        For Each inputVector As List(Of Double) In inputs
            Dim outputVector As List(Of Double) = New List(Of Double)(outputSize)

            For j As Integer = 0 To outputSize - 1
                Dim sum As Double = 0

                For i As Integer = 0 To inputVector.Count - 1
                    sum += inputVector(i) * weights(i)(j)
                Next

                outputVector.Add(sum)
            Next

            output.Add(outputVector)
        Next

        Return output
    End Function

    Public Class LayerNormalization
        Private ReadOnly epsilon As Double
        Private ReadOnly hiddenSize As Integer

        Public Sub New(hiddenSize As Integer, Optional epsilon As Double = 0.000001)
            Me.hiddenSize = hiddenSize
            Me.epsilon = epsilon
        End Sub

        Public Function Normalize(inputs As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim normalizedOutputs As List(Of List(Of Double)) = New List(Of List(Of Double))(inputs.Count)

            For Each inputVector As List(Of Double) In inputs
                Dim mean As Double = inputVector.Average()
                Dim variance As Double = inputVector.Select(Function(x) (x - mean) * (x - mean)).Sum() / hiddenSize
                Dim stdDev As Double = Math.Sqrt(variance + epsilon)

                Dim normalizedVector As List(Of Double) = inputVector.Select(Function(x) (x - mean) / stdDev).ToList()
                normalizedOutputs.Add(normalizedVector)
            Next

            Return normalizedOutputs
        End Function
    End Class

End Class
<Serializable>
Public Class NgramLanguageModel

    Public ngramEncodings As Dictionary(Of String, Integer)

    Public ngramModel As Dictionary(Of String, Dictionary(Of String, Integer))

    Public ngramSize As Integer

    Private ReadOnly rand As Random

    Public Sub New(n As Integer)
        ngramModel = New Dictionary(Of String, Dictionary(Of String, Integer))()
        ngramSize = n
        ngramEncodings = New Dictionary(Of String, Integer)()
        rand = New Random()

    End Sub

    Public ReadOnly Property NgramOrder As Integer
        Get
            Return ngramSize - 1
        End Get
    End Property

    Public Shared Function CalculateProbability(ngramModel As NgramLanguageModel, prediction As String) As Double
        Dim tokens As String() = prediction.Split(" "c)
        Dim probability As Double = 1.0


        For i As Integer = 0 To tokens.Length - 2
            Dim context As String = ngramModel.GetContext(tokens, i)
            Dim nextToken As String = tokens(i + 1)

            If ngramModel.ngramModel.ContainsKey(context) Then
                Dim ngramCounts As Dictionary(Of String, Integer) = ngramModel.ngramModel(context)
                Dim totalOccurrences As Integer = ngramCounts.Values.Sum()

                If ngramCounts.ContainsKey(nextToken) Then
                    Dim count As Integer = ngramCounts(nextToken)
                    Dim tokenProbability As Double = count / totalOccurrences
                    probability *= tokenProbability
                Else
                    probability = 0.0
                    Exit For
                End If
            Else
                probability = 0.0
                Exit For
            End If
        Next

        Return probability
    End Function


    Public Sub AddDocument(doc As String)
        Dim words As String() = PreprocessText(doc)
        Dim numWords As Integer = words.Length - ngramSize

        For i As Integer = 0 To numWords
            Dim currentNgram As String = String.Join(" ", words, i, ngramSize)
            Dim nextWord As String = words(i + ngramSize)

            If Not ngramModel.ContainsKey(currentNgram) Then
                ngramModel(currentNgram) = New Dictionary(Of String, Integer)()
            End If

            If Not ngramModel(currentNgram).ContainsKey(nextWord) Then
                ngramModel(currentNgram)(nextWord) = 0
            End If

            ngramModel(currentNgram)(nextWord) += 1
        Next
    End Sub

    Public Sub AddDocuments(ByRef Docs As List(Of String))
        For Each item In Docs
            Me.AddDocument(item)
        Next
    End Sub

    Public Sub AddNgram(ngram As String)
        ngramModel(ngram) = New Dictionary(Of String, Integer)()
    End Sub

    Public Sub CreateEncodedModel(corpus As String)
        Dim words As String() = PreprocessText(corpus)
        Dim numWords As Integer = words.Length - ngramSize
        Dim position As Integer = 0

        For i As Integer = 0 To numWords
            Dim currentNgram As String = String.Join(" ", words, i, ngramSize)
            Dim nextWord As String = words(i + ngramSize)

            If Not ngramModel.ContainsKey(currentNgram) Then
                ngramModel(currentNgram) = New Dictionary(Of String, Integer)()
            End If

            If Not ngramModel(currentNgram).ContainsKey(nextWord) Then
                ngramModel(currentNgram)(nextWord) = 0
            End If

            ngramModel(currentNgram)(nextWord) += 1

            If Not ngramEncodings.ContainsKey(currentNgram) Then
                ngramEncodings(currentNgram) = position
                position += 1
            End If
        Next
    End Sub

    Public Sub CreateModel(corpus As String)
        Dim words As String() = PreprocessText(corpus)
        Dim numWords As Integer = words.Length - ngramSize

        For i As Integer = 0 To numWords
            Dim currentNgram As String = String.Join(" ", words, i, ngramSize)
            Dim nextWord As String = words(i + ngramSize)

            If Not ngramModel.ContainsKey(currentNgram) Then
                ngramModel(currentNgram) = New Dictionary(Of String, Integer)()
            End If

            If Not ngramModel(currentNgram).ContainsKey(nextWord) Then
                ngramModel(currentNgram)(nextWord) = 0
            End If

            ngramModel(currentNgram)(nextWord) += 1
        Next
    End Sub

    Public Sub CreateModel(documents As List(Of String))
        For Each document In documents
            AddDocument(document)
        Next
    End Sub

    Public Function EstimateProbability(nGramPrefix As String, word As String) As Double
        If ngramModel.ContainsKey(nGramPrefix) AndAlso ngramModel(nGramPrefix).ContainsKey(word) Then
            Dim nGramCount = ngramModel(nGramPrefix)(word)
            Dim totalCount = ngramModel(nGramPrefix).Values.Sum()
            Return nGramCount / totalCount
        End If

        Return 0.0
    End Function

    Public Function GenerateNextWord(nGramPrefix As String) As String
        If ngramModel.ContainsKey(nGramPrefix) Then
            Dim nGramCounts = ngramModel(nGramPrefix)
            Dim totalOccurrences = nGramCounts.Values.Sum()

            Dim randValue = rand.NextDouble()
            Dim cumulativeProb = 0.0

            For Each kvp In nGramCounts
                cumulativeProb += kvp.Value / totalOccurrences
                If cumulativeProb >= randValue Then
                    Return kvp.Key
                End If
            Next
        End If

        Return ""
    End Function

    Public Overridable Function GenerateText(seedPhrase As String, length As Integer) As String
        Dim generatedText As List(Of String) = seedPhrase.Split(" "c).ToList()

        For i As Integer = 0 To length - ngramSize
            Dim nGramPrefix = String.Join(" ", generatedText.Skip(i).Take(ngramSize - 1))
            Dim nextWord = GenerateNextWord(nGramPrefix)
            generatedText.Add(nextWord)
        Next

        Return String.Join(" ", generatedText)
    End Function

    Public Overridable Function GenerateText(maxLength As Integer, seedPhrase As String) As String
        Dim tokens As List(Of String) = New List(Of String)(seedPhrase.Split(" "c))

        While tokens.Count < maxLength
            Dim context As String = GetContextfrom(tokens.ToArray(), tokens.Count - 1)

            If ngramModel.ContainsKey(context) Then
                Dim ngramCounts As Dictionary(Of String, Integer) = ngramModel(context)
                Dim totalOccurrences As Integer = ngramCounts.Values.Sum()
                Dim randomNumber As Double = New Random().NextDouble()
                Dim cumulativeProbability As Double = 0.0

                For Each tokenCount As KeyValuePair(Of String, Integer) In ngramCounts
                    Dim tokenProbability As Double = tokenCount.Value / totalOccurrences
                    cumulativeProbability += tokenProbability

                    If cumulativeProbability >= randomNumber Then
                        tokens.Add(tokenCount.Key)
                        Exit For
                    End If
                Next
            Else
                Exit While
            End If
        End While

        Return String.Join(" ", tokens)
    End Function


    Public Function GetCount(ngram As String) As Integer

        For Each item In ngramEncodings
            If item.Key = ngram Then
                Return ngramEncodings(ngram)
            End If
        Next

        Return 0
    End Function

    Public Function GetEncoding(currentNgram As String) As Integer
        Dim position As Integer = GetPosition(currentNgram)
        Return position
    End Function

    Public Function GetNextToken(context As String) As String
        Dim nextToken As String = ""

        If ngramModel.ContainsKey(context) Then
            Dim ngramCounts As Dictionary(Of String, Integer) = ngramModel(context)
            nextToken = ngramCounts.OrderByDescending(Function(ngram) ngram.Value).FirstOrDefault().Key
        End If

        Return nextToken
    End Function

    Public Function GetNgrams() As String()
        Return ngramModel.Keys.ToArray()
    End Function

    Public Function GetPosition(currentNgram As String) As Integer
        If ngramEncodings.ContainsKey(currentNgram) Then
            Return ngramEncodings(currentNgram)
        End If

        Return -1
    End Function

    Public Function GetProbability(ngram As String) As Double
        Return GetCount(ngram) / ngramModel.Values.SelectMany(Function(dict) dict.Values).Sum()
    End Function

    Public Function GetProbability(currentNgram As String, nextWord As String) As Double
        If ngramModel.ContainsKey(currentNgram) AndAlso ngramModel(currentNgram).ContainsKey(nextWord) Then
            Dim totalCount As Integer = ngramModel(currentNgram).Values.Sum()
            Dim ngramCount As Integer = ngramModel(currentNgram)(nextWord)
            Return CDbl(ngramCount) / totalCount
        End If

        Return 0.0
    End Function

    Public Function GetRandomNgram() As String
        Dim random As New Random()
        Dim ngrams As String() = ngramModel.Keys.ToArray()
        Dim randomIndex As Integer = random.Next(ngrams.Length)
        Return ngrams(randomIndex)
    End Function

    Public Function getTokens(Query As String) As List(Of String)
        Dim tokens As New List(Of String)
        Dim Tok = Split(Query, " ")
        For Each item In Tok
            tokens.Add(item)
        Next
        Return tokens
    End Function


    Public Function LookupNgram(ngram As String) As Integer
        If ngramModel.ContainsKey(ngram) Then
            Return ngramModel(ngram).Values.Sum()
        End If
        Return 0
    End Function


    Public Function PredictNextWord(currentNgram As String) As String
        If ngramModel.ContainsKey(currentNgram) Then
            Dim nextWords As Dictionary(Of String, Integer) = ngramModel(currentNgram)
            Return nextWords.OrderByDescending(Function(x) x.Value).FirstOrDefault().Key
        End If

        Return ""
    End Function


    Public Function PreprocessText(text As String) As String()
        ' Preprocess the text by removing unnecessary characters and converting to lowercase
        text = text.ToLower()
        text = text.Replace(".", " .")
        text = text.Replace(",", " ,")
        text = text.Replace(";", " ;")
        text = text.Replace(":", " :")
        text = text.Replace("!", " !")
        text = text.Replace("?", " ?")

        ' Split the text into words
        Return text.Split(New Char() {" "c}, StringSplitOptions.RemoveEmptyEntries)
    End Function

    Public Sub RemoveDocument(doc As String)
        Dim words As String() = PreprocessText(doc)
        Dim numWords As Integer = words.Length - ngramSize

        For i As Integer = 0 To numWords
            Dim currentNgram As String = String.Join(" ", words, i, ngramSize)
            Dim nextWord As String = words(i + ngramSize)

            If ngramModel.ContainsKey(currentNgram) Then
                Dim nextWords As Dictionary(Of String, Integer) = ngramModel(currentNgram)
                If nextWords.ContainsKey(nextWord) Then
                    nextWords(nextWord) -= 1
                    If nextWords(nextWord) <= 0 Then
                        nextWords.Remove(nextWord)
                    End If
                End If
            End If
        Next
    End Sub

    Public Sub RemoveNgram(ngram As String)
        ngramModel.Remove(ngram)
    End Sub

    Public Overridable Sub Train(corpus As List(Of String))
        For Each sentence In corpus
            Dim words = sentence.Split(" "c)
            For i As Integer = 0 To words.Length - ngramSize
                Dim nGramPrefix = String.Join(" ", words, i, ngramSize - 1)
                Dim nGramSuffix = words(i + ngramSize - 1)

                If Not ngramModel.ContainsKey(nGramPrefix) Then
                    ngramModel(nGramPrefix) = New Dictionary(Of String, Integer)()
                End If

                If Not ngramModel(nGramPrefix).ContainsKey(nGramSuffix) Then
                    ngramModel(nGramPrefix)(nGramSuffix) = 0
                End If

                ngramModel(nGramPrefix)(nGramSuffix) += 1
            Next
        Next
        For Each line In corpus
            Dim tokens = line.Split()
            For i As Integer = 0 To tokens.Length - NgramOrder
                Dim context As String = GetContext(tokens, i)
                Dim nextToken As String = tokens(i + NgramOrder)
                UpdateNgramModel(context, nextToken)
            Next
        Next
    End Sub


    Public Function UpdateNgram(oldNgram As String, newNgram As String) As Boolean
        If ngramModel.ContainsKey(oldNgram) AndAlso Not ngramModel.ContainsKey(newNgram) Then
            ' Update ngramModel
            ngramModel(newNgram) = ngramModel(oldNgram)
            ngramModel.Remove(oldNgram)

            ' Update ngramEncodings
            If ngramEncodings.ContainsKey(oldNgram) Then
                Dim position As Integer = ngramEncodings(oldNgram)
                ngramEncodings.Remove(oldNgram)
                ngramEncodings(newNgram) = position
            End If

            Return True
        End If
        Return False
    End Function

    Public Shared Function GetContextfrom(tokens As String(), index As Integer) As String
        Return String.Join(" ", tokens.Take(index + 1))
    End Function

    Public Function GetContext(tokens As List(Of String)) As String
        Dim contextTokens As List(Of String) = tokens.Skip(Math.Max(0, tokens.Count - NgramOrder)).ToList()
        Return String.Join(" ", contextTokens)
    End Function

    Public Function GetContext(tokens As String(), index As Integer) As String
        Dim contextTokens As New List(Of String)()
        For i As Integer = index To index + NgramOrder - 1
            contextTokens.Add(tokens(i))
        Next
        Return String.Join(" ", contextTokens)
    End Function

    Private Sub UpdateNgramModel(context As String, nextToken As String)
        If Not ngramModel.ContainsKey(context) Then
            ngramModel.Add(context, New Dictionary(Of String, Integer)())
        End If

        Dim ngramCounts As Dictionary(Of String, Integer) = ngramModel(context)
        If ngramCounts.ContainsKey(nextToken) Then
            ngramCounts(nextToken) += 1
        Else
            ngramCounts.Add(nextToken, 1)
        End If
    End Sub
End Class
<Serializable>
Public Class PositionalDecoder
    ''' <summary>
    ''' Only a list of the vocabulary words (to create an index) 
    ''' this should be the same list used to encode (Must be Set)
    ''' </summary>
    Public Vocabulary As New List(Of String)

    Private ReadOnly decodingMatrix As List(Of List(Of Double))
    Public Sub New(maxLength As Integer, embeddingSize As Integer, ByRef vocab As List(Of String))
        decodingMatrix = New List(Of List(Of Double))()
        Vocabulary = vocab
        ' Create the decoding matrix
        For pos As Integer = 0 To maxLength - 1
            Dim decodingRow As List(Of Double) = New List(Of Double)()

            For i As Integer = 0 To embeddingSize - 1
                Dim angle As Double = pos / Math.Pow(10000, (2 * i) / embeddingSize)
                decodingRow.Add(Math.Sin(angle))
                decodingRow.Add(Math.Cos(angle))
            Next

            decodingMatrix.Add(decodingRow)
        Next
    End Sub
    Public Function Decode(encodedInputs As List(Of List(Of Double))) As List(Of String)
        Dim decodedTokens As List(Of String) = New List(Of String)()

        For Each encoding As List(Of Double) In encodedInputs
            ' Retrieve the token index based on the encoding
            Dim tokenIndex As Integer = GetTokenIndex(encoding)

            ' Retrieve the token based on the index
            If tokenIndex >= 0 Then
                Dim token As String = GetToken(tokenIndex)
                decodedTokens.Add(token)
            Else
                ' Handle unknown encodings if necessary
            End If
        Next

        Return decodedTokens
    End Function
    Public Function iDecode(encodedInputs As List(Of List(Of Double))) As List(Of String)
        Dim decodedTokens As List(Of String) = New List(Of String)()

        For Each encoding As List(Of Double) In encodedInputs
            ' Retrieve the token index based on the encoding
            Dim tokenIndex As Integer = GetTokenIndex(encoding)

            ' Retrieve the token based on the index
            If tokenIndex >= 0 Then
                Dim token As String = GetToken(tokenIndex)
                decodedTokens.Add(token)
            Else
                ' Handle unknown encodings if necessary
            End If
        Next

        Return decodedTokens
    End Function

    Private Function GetToken(tokenIndex As Integer) As String
        ' Retrieve the token based on the index
        ' For simplicity, let's assume a fixed vocabulary
        Dim vocabulary As List(Of String) = GetVocabulary()

        If tokenIndex >= 0 AndAlso tokenIndex < vocabulary.Count Then
            Return vocabulary(tokenIndex)
        Else
            Return "Unknown" ' Unknown token
        End If
    End Function

    Public Function GetTokenIndex(encoding As List(Of Double)) As Integer
        ' Retrieve the index of the token based on the encoding
        ' For simplicity, let's assume a fixed vocabulary
        Dim vocabulary As List(Of String) = GetVocabulary()

        For i As Integer = 0 To decodingMatrix.Count - 1
            If encoding.SequenceEqual(decodingMatrix(i)) Then
                Return i
            End If
        Next

        Return -1 ' Token not found
    End Function
    Private Function GetVocabulary() As List(Of String)
        ' Return the vocabulary list
        ' Modify this function as per your specific vocabulary
        Return Vocabulary
    End Function
End Class
<Serializable>
Public Structure Vocabulary
    Private iValues As List(Of Token)
    Public ReadOnly Property Values As List(Of Token)
        Get
            Return iValues
        End Get
    End Property
    Public Structure Token
        Public Text
        Public Vocabulary_ID As Integer
        Public Encoding As Integer
    End Structure
    Private iVocabList As List(Of String)
    Public ReadOnly Property VocabList As List(Of String)
        Get
            Return iVocabList
        End Get
    End Property

    Public Function ADD_NEW(ByRef Token As String) As Boolean
        If CheckExists(Token) = False Then
            Dim NewTok As New Token
            NewTok.Text = Token
            NewTok.Vocabulary_ID = VocabList.Count + 1
            iValues.Add(NewTok)
            Return True
        Else
            Return False
        End If
    End Function
    Public Function ADD_NEW(ByRef Token As String, Encoding As Integer) As Boolean
        If CheckExists(Token) = False Then
            Dim NewTok As New Token
            NewTok.Text = Token
            NewTok.Vocabulary_ID = VocabList.Count + 1
            NewTok.Encoding = Encoding
            iValues.Add(NewTok)
            Return True
        Else
            Return False
        End If
    End Function
    Public Function LOOKUP(ByRef Query As Integer) As String

        If CheckExists(Query) = True Then Return VocabList(Query)


        Return "Not Found"
    End Function
    Public Function CheckExists(ByRef Query As String) As Boolean
        Return VocabList.Contains(Query)
    End Function
    Private Function CheckExists(ByRef Query As Integer) As Boolean
        Return VocabList.Count < Query
    End Function
End Structure
''' <summary>
''' These are the options of transfer functions available to the network
''' This is used to select which function to be used:
''' The derivative function can also be selected using this as a marker
''' </summary>
Public Enum TransferFunctionType
    none
    sigmoid
    HyperbolTangent
    BinaryThreshold
    RectifiedLinear
    Logistic
    StochasticBinary
    Gaussian
    Signum
End Enum

''' <summary>
''' Transfer Function used in the calculation of the following layer
''' </summary>
Public Structure TransferFunction

    ''' <summary>
    ''' Returns a result from the transfer function indicated ; Non Derivative
    ''' </summary>
    ''' <param name="TransferFunct">Indicator for Transfer function selection</param>
    ''' <param name="Input">Input value for node/Neuron</param>
    ''' <returns>result</returns>
    Public Shared Function EvaluateTransferFunct(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
        EvaluateTransferFunct = 0
        Select Case TransferFunct
            Case TransferFunctionType.none
                Return Input
            Case TransferFunctionType.sigmoid
                Return Sigmoid(Input)
            Case TransferFunctionType.HyperbolTangent
                Return HyperbolicTangent(Input)
            Case TransferFunctionType.BinaryThreshold
                Return BinaryThreshold(Input)
            Case TransferFunctionType.RectifiedLinear
                Return RectifiedLinear(Input)
            Case TransferFunctionType.Logistic
                Return Logistic(Input)
            Case TransferFunctionType.Gaussian
                Return Gaussian(Input)
            Case TransferFunctionType.Signum
                Return Signum(Input)
        End Select
    End Function

    ''' <summary>
    ''' Returns a result from the transfer function indicated ; Non Derivative
    ''' </summary>
    ''' <param name="TransferFunct">Indicator for Transfer function selection</param>
    ''' <param name="Input">Input value for node/Neuron</param>
    ''' <returns>result</returns>
    Public Shared Function EvaluateTransferFunctionDerivative(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
        EvaluateTransferFunctionDerivative = 0
        Select Case TransferFunct
            Case TransferFunctionType.none
                Return Input
            Case TransferFunctionType.sigmoid
                Return SigmoidDerivitive(Input)
            Case TransferFunctionType.HyperbolTangent
                Return HyperbolicTangentDerivative(Input)
            Case TransferFunctionType.Logistic
                Return LogisticDerivative(Input)
            Case TransferFunctionType.Gaussian
                Return GaussianDerivative(Input)
        End Select
    End Function

    ''' <summary>
    ''' the step function rarely performs well except in some rare cases with (0,1)-encoded
    ''' binary data.
    ''' </summary>
    ''' <param name="Value"></param>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Private Shared Function BinaryThreshold(ByRef Value As Double) As Double

        ' Z = Bias+ (Input*Weight)
        'TransferFunction
        'If Z > 0 then Y = 1
        'If Z < 0 then y = 0

        Return If(Value < 0 = True, 0, 1)
    End Function

    Private Shared Function Gaussian(ByRef x As Double) As Double
        Gaussian = Math.Exp((-x * -x) / 2)
    End Function

    Private Shared Function GaussianDerivative(ByRef x As Double) As Double
        GaussianDerivative = Gaussian(x) * (-x / (-x * -x))
    End Function

    Private Shared Function HyperbolicTangent(ByRef Value As Double) As Double
        ' TanH(x) = (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x))

        Return Math.Tanh(Value)
    End Function

    Private Shared Function HyperbolicTangentDerivative(ByRef Value As Double) As Double
        HyperbolicTangentDerivative = 1 - (HyperbolicTangent(Value) * HyperbolicTangent(Value)) * Value
    End Function

    'Linear Neurons
    ''' <summary>
    ''' in a liner neuron the weight(s) represent unknown values to be determined the
    ''' outputs could represent the known values of a meal and the inputs the items in the
    ''' meal and the weights the prices of the individual items There are no hidden layers
    ''' </summary>
    ''' <remarks>
    ''' answers are determined by determining the weights of the linear neurons the delta
    ''' rule is used as the learning rule: Weight = Learning rate * Input * LocalError of neuron
    ''' </remarks>
    Private Shared Function Linear(ByRef value As Double) As Double
        ' Output = Bias + (Input*Weight)
        Return value
    End Function

    'Non Linear neurons
    Private Shared Function Logistic(ByRef Value As Double) As Double
        'z = bias + (sum of all inputs ) * (input*weight)
        'output = Sigmoid(z)
        'derivative input = z/weight
        'derivative Weight = z/input
        'Derivative output = output*(1-Output)
        'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

        Return 1 / 1 + Math.Exp(-Value)
    End Function

    Private Shared Function LogisticDerivative(ByRef Value As Double) As Double
        'z = bias + (sum of all inputs ) * (input*weight)
        'output = Sigmoid(z)
        'derivative input = z/weight
        'derivative Weight = z/input
        'Derivative output = output*(1-Output)
        'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

        Return Logistic(Value) * (1 - Logistic(Value))
    End Function

    Private Shared Function RectifiedLinear(ByRef Value As Double) As Double
        'z = B + (input*Weight)
        'If Z > 0 then output = z
        'If Z < 0 then output = 0
        If Value < 0 = True Then

            Return 0
        Else
            Return Value
        End If
    End Function

    ''' <summary>
    ''' the log-sigmoid function constrains results to the range (0,1), the function is
    ''' sometimes said to be a squashing function in neural network literature. It is the
    ''' non-linear characteristics of the log-sigmoid function (and other similar activation
    ''' functions) that allow neural networks to model complex data.
    ''' </summary>
    ''' <param name="Value"></param>
    ''' <returns></returns>
    ''' <remarks>1 / (1 + Math.Exp(-Value))</remarks>
    Private Shared Function Sigmoid(ByRef Value As Integer) As Double
        'z = Bias + (Input*Weight)
        'Output = 1/1+e**z
        Return 1 / (1 + Math.Exp(-Value))
    End Function

    Private Shared Function SigmoidDerivitive(ByRef Value As Integer) As Double
        Return Sigmoid(Value) * (1 - Sigmoid(Value))
    End Function

    Private Shared Function Signum(ByRef Value As Integer) As Double
        'z = Bias + (Input*Weight)
        'Output = 1/1+e**z
        Return Math.Sign(Value)
    End Function

    Private Shared Function StochasticBinary(ByRef value As Double) As Double
        'Uncreated
        Return value
    End Function

End Structure
Public Class Softmax
    Public Shared Function Softmax(matrix2 As Integer(,)) As Double(,)
        Dim numRows As Integer = matrix2.GetLength(0)
        Dim numColumns As Integer = matrix2.GetLength(1)

        Dim softmaxValues(numRows - 1, numColumns - 1) As Double

        ' Compute softmax values for each row
        For i As Integer = 0 To numRows - 1
            Dim rowSum As Double = 0

            ' Compute exponential values and sum of row elements
            For j As Integer = 0 To numColumns - 1
                softmaxValues(i, j) = Math.Sqrt(Math.Exp(matrix2(i, j)))
                rowSum += softmaxValues(i, j)
            Next

            ' Normalize softmax values for the row
            For j As Integer = 0 To numColumns - 1
                softmaxValues(i, j) /= rowSum
            Next
        Next

        ' Display the softmax values
        Console.WriteLine("Calculated:" & vbNewLine)
        For i As Integer = 0 To numRows - 1
            For j As Integer = 0 To numColumns - 1

                Console.Write(softmaxValues(i, j).ToString("0.0000") & " ")
            Next
            Console.WriteLine(vbNewLine & "---------------------")
        Next
        Return softmaxValues
    End Function
    Public Shared Sub Main()
        Dim input() As Double = {1.0, 2.0, 3.0}

        Dim output() As Double = Softmax(input)

        Console.WriteLine("Input: {0}", String.Join(", ", input))
        Console.WriteLine("Softmax Output: {0}", String.Join(", ", output))
        Console.ReadLine()
    End Sub

    Public Shared Function Softmax(ByVal input() As Double) As Double()
        Dim maxVal As Double = input.Max()

        Dim exponentiated() As Double = input.Select(Function(x) Math.Exp(x - maxVal)).ToArray()

        Dim sum As Double = exponentiated.Sum()

        Dim softmaxOutput() As Double = exponentiated.Select(Function(x) x / sum).ToArray()

        Return softmaxOutput
    End Function
End Class
Public Class SimilarityCalculator


    Public Shared Function CalculateCosineSimilarity(vector1 As List(Of Double), vector2 As List(Of Double)) As Double
        If vector1.Count <> vector2.Count Then
            Throw New ArgumentException("Vector dimensions do not match.")
        End If

        Dim dotProduct As Double = 0
        Dim magnitude1 As Double = 0
        Dim magnitude2 As Double = 0

        For i As Integer = 0 To vector1.Count - 1
            dotProduct += vector1(i) * vector2(i)
            magnitude1 += Math.Pow(vector1(i), 2)
            magnitude2 += Math.Pow(vector2(i), 2)
        Next

        magnitude1 = Math.Sqrt(magnitude1)
        magnitude2 = Math.Sqrt(magnitude2)

        Return dotProduct / (magnitude1 * magnitude2)
    End Function

    Public Shared Function CalculateJaccardSimilarity(sentences1 As List(Of String), sentences2 As List(Of String)) As Double
        Dim set1 As New HashSet(Of String)(sentences1)
        Dim set2 As New HashSet(Of String)(sentences2)

        Return SimilarityCalculator.CalculateJaccardSimilarity(set1, set2)
    End Function

    Public Shared Function CalculateJaccardSimilarity(set1 As HashSet(Of String), set2 As HashSet(Of String)) As Double
        Dim intersectionCount As Integer = set1.Intersect(set2).Count()
        Dim unionCount As Integer = set1.Union(set2).Count()

        Return CDbl(intersectionCount) / CDbl(unionCount)
    End Function

End Class
<Serializable>
Public Class Perceptron

    Public Property Weights As Double() ' The weights of the perceptron

    Private Function Sigmoid(x As Double) As Double ' The sigmoid activation function

        Return 1 / (1 + Math.Exp(-x))
    End Function

    ''' <summary>
    ''' the step function rarely performs well except in some rare cases with (0,1)-encoded
    ''' binary data.
    ''' </summary>
    ''' <param name="Value"></param>
    ''' <returns></returns>
    ''' <remarks></remarks>
    Private Shared Function BinaryThreshold(ByRef Value As Double) As Double

        ' Z = Bias+ (Input*Weight)
        'TransferFunction
        'If Z > 0 then Y = 1
        'If Z < 0 then y = 0

        Return If(Value < 0 = True, 0, 1)
    End Function



    Public Sub New(NumberOfInputs As Integer) ' Constructor that initializes the weights and bias of the perceptron
        CreateWeights(NumberOfInputs)

    End Sub

    Public Sub CreateWeights(NumberOfInputs As Integer) ' Constructor that initializes the weights and bias of the perceptron
        Weights = New Double(NumberOfInputs - 1) {}
        For i As Integer = 0 To NumberOfInputs - 1
            Weights(i) = Rnd(1.0)
        Next

    End Sub

    ' Function to calculate output
    Public Function ForwardLinear(inputs As Double()) As Integer
        Dim sum = 0.0

        ' Loop through inputs and calculate sum of weights times inputs
        For i = 0 To inputs.Length - 1
            sum += inputs(i)
        Next

        Return sum
    End Function
    Public Function Forward(inputs As Double()) As Integer
        Dim sum = 0.0

        ' Loop through inputs and calculate sum of weights times inputs
        For i = 0 To inputs.Length - 1
            sum += Weights(i) * inputs(i)
        Next

        Return sum
    End Function
    Public Function ForwardSigmoid(inputs As Double()) As Double ' Compute the output of the perceptron given an input
        CreateWeights(inputs.Count)
        Dim sum As Double = 0
        'Collect the sum of the inputs * Weight
        For i As Integer = 0 To inputs.Length - 1
            sum += inputs(i) * Weights(i)
        Next

        'Activate
        'We Return the sigmoid of the sum to produce the output
        Return Sigmoid(sum)
    End Function

    Public Function ForwardBinaryThreshold(inputs As Double()) As Double ' Compute the output of the perceptron given an input
        CreateWeights(inputs.Count)
        Dim sum As Double = 0 ' used to hold the output

        'Collect the sum of the inputs * Weight
        For i As Integer = 0 To inputs.Length - 1
            sum += inputs(i) * Weights(i)
        Next

        'Activate
        'We Return the sigmoid of the sum to produce the output , Applying the Binary threshold funciton to it
        Return BinaryThreshold(Sigmoid(sum))
    End Function

    ' Function to train the perceptron
    Public Sub Train(inputs As Double(), desiredOutput As Integer, threshold As Double, MaxEpochs As Integer, LearningRate As Double)
        Dim guess = Forward(inputs)
        Dim nError As Integer = 0
        Dim CurrentEpoch = 0

        Do Until threshold < nError Or
                        CurrentEpoch = MaxEpochs
            CurrentEpoch += 1

            nError = desiredOutput - guess

            ' Loop through inputs and update weights based on error and learning rate
            For i = 0 To inputs.Length - 1
                _Weights(i) += LearningRate * nError * inputs(i)
            Next

        Loop

    End Sub

End Class
Public Class eXamples
    Public Sub TokenEncoding()
        ' Create a Vocabulary instance
        Dim vocab As New Vocabulary()

        ' Add tokens to the vocabulary
        vocab.ADD_NEW("cat", 0)
        vocab.ADD_NEW("dog", 1)
        vocab.ADD_NEW("fish", 2)

        ' Look up the encoding for a token
        Dim catEncoding As Integer = vocab.Values.Find(Function(t) t.Text = "cat").Encoding
        Console.WriteLine("Encoding for 'cat': " & catEncoding) ' Output: Encoding for 'cat': 0
    End Sub
    Public Sub PositionalCoding()
        ' Create a PositionalDecoder instance
        Dim maxLength As Integer = 10
        Dim embeddingSize As Integer = 16
        Dim vocabList As List(Of String) = New List(Of String)({"cat", "dog", "fish"})
        Dim decoder As New PositionalDecoder(maxLength, embeddingSize, vocabList)

        ' Sample encoded inputs
        Dim encodedInputs As New List(Of List(Of Double))()
        encodedInputs.Add(New List(Of Double)({0.5, 0.866}))
        encodedInputs.Add(New List(Of Double)({-0.866, 0.5}))

        ' Decode the encoded inputs
        Dim decodedTokens As List(Of String) = decoder.Decode(encodedInputs)
        Console.WriteLine("Decoded Tokens: " & String.Join(", ", decodedTokens)) ' Output: Decoded Tokens: cat, dog

    End Sub
    Public Class TransformerTranslationModelExample
        Inherits NgramLanguageModel
        Dim PositionalEncoder As PositionalEncoding
        Dim Positionaldecoder As PositionalDecoder

        Dim Vocabulary As List(Of String)
        ' Initialize the model with required parameters
        Public Sub New(vocab As List(Of String), maxSeqLength As Integer, embeddingSize As Integer)
            MyBase.New(3) ' Assuming you want trigram models
            Vocabulary = vocab
            ' Create a PositionalDecoder instance for decoding embeddings
            PositionalEncoder = New PositionalEncoding(maxSeqLength, embeddingSize, Vocabulary)
            Positionaldecoder = New PositionalDecoder(maxSeqLength, embeddingSize, Vocabulary)
        End Sub

        ' Translate input text using the transformer model
        Public Function Translate(inputText As String, sourceLanguage As String, targetLanguage As String) As String
            ' Tokenize the input text
            Dim sourceTokens As List(Of String) = Tokenize(inputText)

            ' Convert source tokens to embeddings
            Dim sourceEmbeddings As List(Of List(Of List(Of Double))) = ConvertTokensToEmbeddings(sourceTokens)

            ' Apply transformer operations (simplified for demonstration)
            Dim translatedEmbeddings As List(Of List(Of Double)) = ApplyTransformer(sourceEmbeddings)

            ' Decode translated embeddings to target language tokens
            Dim translatedTokens As List(Of String) = DecodeEmbeddings(translatedEmbeddings)

            ' Join translated tokens to form translated text
            Dim translatedText As String = String.Join(" ", translatedTokens)

            Return translatedText
        End Function

        ' Tokenize input text
        Private Function Tokenize(text As String) As List(Of String)
            Return PreprocessText(text).ToList
        End Function

        ' Convert tokens to embeddings
        Private Function ConvertTokensToEmbeddings(tokens As List(Of String)) As List(Of List(Of List(Of Double)))
            Dim embeddings As New List(Of List(Of List(Of Double)))
            For Each token In tokens
                Dim encoding = Vocabulary.IndexOf(token)
                embeddings.Add(PositionalEncoder.Encode(tokens))
            Next
            Return embeddings
        End Function

        ' Apply transformer operations (simplified for demonstration)
        Private Function ApplyTransformer(embeddings As List(Of List(Of List(Of Double)))) As List(Of List(Of Double))
            ' Simplified transformer operations (replace with actual transformer logic)
            Dim translatedEmbeddings As New List(Of List(Of Double))
            '

            For Each Entry In embeddings

            Next
            'To be Implemented


            '
            Return translatedEmbeddings
        End Function

        ' Decode embeddings to target language tokens
        Private Function DecodeEmbeddings(embeddings As List(Of List(Of Double))) As List(Of String)
            Dim decodedTokens As New List(Of String)
            For Each embedding In embeddings
                Dim tokenIndex As Integer = Positionaldecoder.GetTokenIndex(embedding)
                If tokenIndex >= 0 Then
                    Dim token As String = Vocabulary(tokenIndex)
                    decodedTokens.Add(token)
                Else
                    ' Handle unknown encodings if necessary
                End If
            Next
            Return decodedTokens
        End Function
    End Class
End Class