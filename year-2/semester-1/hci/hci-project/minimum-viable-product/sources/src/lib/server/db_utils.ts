import type { t } from "$lib/server/db";
import { decodeBase64urlIgnorePadding, encodeBase64urlNoPadding } from "@oslojs/encoding";
import { isNotNull } from "drizzle-orm";

export function generateId() {
    return Buffer.from(crypto.getRandomValues(new Uint8Array(16)));
}

export function encodeId(id: Buffer) {
    return encodeBase64urlNoPadding(id);
}

export function decodeId(id: string) {
    return Buffer.from(decodeBase64urlIgnorePadding(id));
}

export function selectUser(table: typeof t.user) {
    return {
        id: table.id,
        displayName: table.displayName,
        avatar: isNotNull(table.avatar).mapWith((x) => !!x),
        avatarModified: table.avatarModified,
        username: table.username
    } as const;
}

export function encodeUser(row: {
    id: Buffer;
    displayName: string;
    avatar: boolean;
    avatarModified: Date;
    username: string;
}) {
    return {
        id: encodeId(row.id),
        displayName: row.displayName,
        avatar: row.avatar,
        avatarModified: row.avatarModified,
        username: row.username
    };
}

export function selectGroup(table: typeof t.group) {
    return {
        id: table.id,
        displayName: table.displayName,
        avatar: isNotNull(table.avatar).mapWith((x) => !!x),
        avatarModified: table.avatarModified,
        adminUserId: table.adminUserId
    };
}

export function encodeGroup(row: {
    id: Buffer;
    displayName: string;
    avatar: boolean;
    avatarModified: Date;
    adminUserId: Buffer | null;
}) {
    return {
        id: encodeId(row.id),
        displayName: row.displayName,
        avatar: row.avatar,
        avatarModified: row.avatarModified,
        adminUserId: row.adminUserId ? encodeId(row.adminUserId) : null
    };
}

export function encodeLoan(row: {
    id: Buffer;
    issuerUserId: Buffer | null;
    groupId: Buffer | null;
    createdOn: Date;
}) {
    return {
        id: encodeId(row.id),
        issuerUserId: row.issuerUserId != null ? encodeId(row.issuerUserId) : null,
        groupId: row.groupId != null ? encodeId(row.groupId) : null,
        createdOn: row.createdOn
    };
}

export function encodeLoanPayment(row: {
    id: Buffer;
    loanId: Buffer;
    userId: Buffer | null;
    value: number;
    valuePaid: number;
    valuePaidConfirmed: number;
}) {
    return {
        id: encodeId(row.id),
        loanId: encodeId(row.loanId),
        userId: row.userId != null ? encodeId(row.userId) : null,
        value: row.value / 100,
        valuePaid: row.valuePaid / 100,
        valuePaidConfirmed: row.valuePaidConfirmed / 100
    };
}
