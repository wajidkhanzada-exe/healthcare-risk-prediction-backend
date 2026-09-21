import { useState } from 'react'
import { supabase } from './supabaseClient'

function Auth({ onLoginSuccess }) {
  const [stage, setStage] = useState('email') // 'email' | 'otp'
  const [email, setEmail] = useState('')
  const [otp, setOtp] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [message, setMessage] = useState(null)

  const handleSendCode = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    const { error: sendError } = await supabase.auth.signInWithOtp({
      email: email.trim(),
    })

    setLoading(false)

    if (sendError) {
      setError(sendError.message)
      return
    }

    setMessage(`A 6-digit code has been sent to ${email}.`)
    setStage('otp')
  }

  const handleVerifyCode = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    const { data, error: verifyError } = await supabase.auth.verifyOtp({
      email: email.trim(),
      token: otp.trim(),
      type: 'email',
    })

    setLoading(false)

    if (verifyError) {
      setError(verifyError.message)
      return
    }

    onLoginSuccess(data.session)
  }

  const handleResendCode = async () => {
    setError(null)
    setMessage(null)
    setLoading(true)

    const { error: sendError } = await supabase.auth.signInWithOtp({
      email: email.trim(),
    })

    setLoading(false)

    if (sendError) {
      setError(sendError.message)
      return
    }

    setMessage(`A new code has been sent to ${email}.`)
  }

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4">
      <div className="w-full max-w-sm bg-white border border-slate-200 rounded-xl shadow-sm p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4.5 12.75l6 6 9-13.5" />
            </svg>
          </div>
          <h1 className="text-xl font-semibold text-slate-900">Health Risk Assessment</h1>
        </div>

        {stage === 'email' && (
          <form onSubmit={handleSendCode}>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Email Address
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm mb-4
                transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-indigo-600 text-white font-medium py-2.5 rounded-lg
                shadow-sm transition-all duration-150
                hover:bg-indigo-700 hover:shadow-md active:scale-[0.99]
                disabled:bg-slate-300 disabled:cursor-not-allowed"
            >
              {loading ? 'Sending...' : 'Send Code'}
            </button>
          </form>
        )}

        {stage === 'otp' && (
          <form onSubmit={handleVerifyCode}>
            <p className="text-sm text-slate-500 mb-4">{message}</p>

            <label className="block text-sm font-medium text-slate-700 mb-1">
              Verification Code
            </label>
            <input
              type="text"
              required
              inputMode="numeric"
              maxLength={6}
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              placeholder="123456"
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm mb-4 tracking-widest text-center text-lg
                transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-indigo-600 text-white font-medium py-2.5 rounded-lg
                shadow-sm transition-all duration-150
                hover:bg-indigo-700 hover:shadow-md active:scale-[0.99]
                disabled:bg-slate-300 disabled:cursor-not-allowed"
            >
              {loading ? 'Verifying...' : 'Verify & Continue'}
            </button>

            <div className="flex items-center justify-between mt-4">
              <button
                type="button"
                onClick={() => { setStage('email'); setOtp(''); setError(null); setMessage(null) }}
                className="text-sm text-slate-500 hover:text-slate-700"
              >
                Change email
              </button>
              <button
                type="button"
                onClick={handleResendCode}
                disabled={loading}
                className="text-sm text-indigo-600 hover:text-indigo-700 font-medium disabled:text-slate-300"
              >
                Resend code
              </button>
            </div>
          </form>
        )}

        {error && (
          <p className="text-rose-600 text-sm mt-4">{error}</p>
        )}
      </div>
    </div>
  )
}

export default Auth