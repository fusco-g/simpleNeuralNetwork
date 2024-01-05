using System.Numerics;

namespace simpleNeuralNetwork.Utilities
{
    public class Matrix<T>(int rows, int cols) where T : IAdditionOperators<T, T, T>,
                                                     IMultiplyOperators<T, T, T>,
                                                     IAdditiveIdentity<T, T>,
                                                     IMultiplicativeIdentity<T, T>,
                                                     ISubtractionOperators<T, T, T>
    {
        #region fields
        public T[,] _data = new T[rows, cols];
        public int _rows = rows;
        public int _cols = cols;
        public bool _isTranspose = false;
        #endregion fields

        #region properties
        public int Rows { get { return !_isTranspose ? _rows : _cols; } }
        public int Cols { get { return !_isTranspose ? _cols : _rows; } }
        #endregion properties

        #region instance methods
        public Matrix<T> Copy()
        {
            return Map(x => x);
        }
        public void Print(Func<T, string>? stringify = null)
        {
            for (int i = 0; i < Rows; i++)
            {
                Console.WriteLine();
                for (int j = 0; j < Cols; j++)
                {
                    Console.Write(" " + (stringify != null ? stringify(this[i, j]) : this[i, j].ToString()));
                }
            }
            Console.WriteLine();
        }

        public Matrix<T> Map(Func<T, T> func)
        {
            var result = new Matrix<T>(_rows, _cols) { _isTranspose = _isTranspose };

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    result[i, j] = func(this[i, j]);
                }
            }

            return result;
        }

        public Matrix<T> ElementwiseMult(Matrix<T> matrix)
        {
            var result = new Matrix<T>(Rows, Cols);

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    result[i, j] = this[i, j] * matrix[i, j];
                }
            }

            return result;
        }
        #endregion instance methods

        #region static methods
        public static Matrix<T> Id(int n) => Id(n, n);
        public static Matrix<T> Id(int rows, int cols)
        {
            var result = new Matrix<T>(rows, cols);
            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Cols; j++)
                {
                    result[i, j] = i == j ? T.MultiplicativeIdentity : T.AdditiveIdentity;
                }
            }
            return result;
        }
        public static Matrix<T> Zeros(int n) => Zeros(n, n);
        public static Matrix<T> Zeros(int rows, int cols)
        {
            var result = new Matrix<T>(rows, cols);
            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Cols; j++)
                {
                    result[i, j] = T.AdditiveIdentity;
                }
            }
            return result;
        }
        #endregion static methods

        #region indexer
        public T this[int i, int j]
        {
            get
            {
                // Check the index boundaries
                if (i >= 0 && i < Rows && j >= 0 && j < Cols)
                    return !_isTranspose ? _data[i, j] : _data[j, i];
                else
                    throw new IndexOutOfRangeException("Index is out of range");
            }
            set
            {
                // Check the index boundaries
                if (i >= 0 && i < Rows && j >= 0 && j < Cols)
                    if (!_isTranspose)
                        _data[i, j] = value;
                    else _data[j, i] = value;
                else
                    throw new IndexOutOfRangeException("Index is out of range");
            }
        }
        #endregion indexer

        #region operators
        public static implicit operator Matrix<T>(T[,] m)
        {
            var result = new Matrix<T>(m.GetLength(0), m.GetLength(1));
            result._data = m;
            return result;
        }

        public static implicit operator Matrix<T>(T[] m)
        {
            var result = new Matrix<T>(m.GetLength(0), 1);

            for (int i = 0; i < result.Rows; i++)
            {
                result[i, 0] = m[i];
            }

            return result;
        }
        public static Matrix<T> operator +(Matrix<T> a, Matrix<T> b)
        {
            ArgumentNullException.ThrowIfNull(a);
            ArgumentNullException.ThrowIfNull(b);
            if (a.Rows != b.Rows || a.Cols != b.Cols) throw new ArgumentException("sizes must be equal");
            var result = new Matrix<T>(a.Rows, a.Cols);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }

            return result;
        }
        public static Matrix<T> operator *(Matrix<T> a, T b)
        {
            ArgumentNullException.ThrowIfNull(a);
            ArgumentNullException.ThrowIfNull(b);

            var result = new Matrix<T>(a.Rows, a.Cols);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] * b;
                }
            }

            return result;
        }
        public static Matrix<T> operator *(T a, Matrix<T> b) => b * a;
        public static Matrix<T> operator *(Matrix<T> a, Matrix<T> b)
        {
            ArgumentNullException.ThrowIfNull(a);
            ArgumentNullException.ThrowIfNull(b);
            if (a.Cols != b.Rows) throw new ArgumentException("sizes must be equal");

            var result = new Matrix<T>(a.Rows, b.Cols);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < b.Cols; j++)
                {
                    for (int k = 0; k < a.Cols; k++)
                    {
                        result[i, j] += a[i, k] * b[k, j];
                    }
                }
            }

            return result;
        }
        public static Matrix<T> operator -(Matrix<T> a)
        {
            var result = new Matrix<T>(a.Rows, a.Cols);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {

                    result[i, j] = T.AdditiveIdentity - a[i, j];

                }
            }

            return result;
        }
        public static Matrix<T> operator -(Matrix<T> a, Matrix<T> b) => a + (-b);
        public static Matrix<T> operator !(Matrix<T> a) => new Matrix<T>(a._rows, a._cols) { _data = a._data, _isTranspose = !a._isTranspose };

        #endregion operators
    }
}
