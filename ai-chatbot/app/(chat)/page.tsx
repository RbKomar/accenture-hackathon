import { nanoid } from '@/lib/utils'
import { Chat } from '@/components/chat'
import { Session } from '@/lib/types'
import { getMissingKeys } from '../actions'

export const metadata = {
  title: 'Next.js AI Chatbot'
}

export default async function IndexPage() {
  const id = nanoid()
  const missingKeys = await getMissingKeys()

  return <Chat id={id} missingKeys={missingKeys} />
}
